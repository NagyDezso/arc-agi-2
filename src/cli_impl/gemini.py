import contextlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from .base import BaseCLI, capture_raw_output_line, find_last_grid

SOLVER_MD = """\
# ARC-AGI Puzzle Solver

Read `task.json`. It has `train` (input/output pairs) and `test` (one test input).
Find the transformation pattern and apply it to the test input.

Use `python3` for scripting. All common scientific/mathematical packages are pre-installed — use whatever you need.

Output grids must contain only integers 0-9.

## Approach
Write `transform.py` with a Python function `transform(grid: np.ndarray) -> np.ndarray`.
The function takes a 2D numpy integer array and returns a 2D numpy integer array.
Test against ALL training pairs. Iterate until correct.

When analyzing, consider: object manipulation, color changes, spatial patterns,
object relationships, grid structure (borders, separators, subgrids).
"""

GEMINI_PRICING = {
    "gemini-3-flash-preview": (0.50, 3.00, 0.05),
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
    "gemini-2.5-pro": (1.25, 10.00, 0.125),
}

_TOOL_NAME_MAP = {
    "run_shell_command": "Bash",
    "read_file": "Read",
    "write_file": "Write",
    "write_new_file": "Write",
    "edit_file": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "list_directory": "Glob",
}


class GeminiCLI(BaseCLI):
    def workspace_extras(self, ws_path: Path):
        gemini_dir = ws_path / ".gemini"
        gemini_dir.mkdir(parents=True, exist_ok=True)
        settings = json.dumps(
            {
                "general": {"enableAutoUpdate": False},
                "model": {
                    "maxSessionTurns": 500,
                    "disableLoopDetection": True,
                },
                "shell": {
                    "inactivityTimeout": 1800,
                },
                "agents": {
                    "overrides": {
                        "generalist": {
                            "runConfig": {
                                "maxTurns": 200,
                                "maxTimeMinutes": 360,
                            }
                        }
                    }
                },
            },
            indent=2,
        )
        # Gemini OAuth initialization
        gemini_access_token = os.environ.get("GEMINI_OAUTH_ACCESS_TOKEN")
        if gemini_access_token:
            (gemini_dir / "oauth_creds.json").write_text(
                json.dumps(
                    {
                        "access_token": gemini_access_token,
                        "refresh_token": os.environ.get("GEMINI_OAUTH_REFRESH_TOKEN"),
                        "scope": (
                            "https://www.googleapis.com/auth/userinfo.email openid "
                            "https://www.googleapis.com/auth/userinfo.profile "
                            "https://www.googleapis.com/auth/cloud-platform"
                        ),
                        "token_type": "Bearer",
                        "id_token": os.environ.get("GEMINI_OAUTH_ID_TOKEN"),
                        "expiry_date": 1772303384460,
                    }
                )
            )
        (gemini_dir / "settings.json").write_text(settings, encoding="utf-8")

    def calculate_cost(self, model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        pricing = GEMINI_PRICING.get(model)
        if pricing is None:
            return 0.0
        input_rate, output_rate, cached_rate = pricing
        return (
            input_tokens * input_rate / 1_000_000
            + cached_tokens * cached_rate / 1_000_000
            + output_tokens * output_rate / 1_000_000
        )

    def run_session(
        self, ws_path: Path, model: str, initial_prompt: str, feedback: str, iteration: int
    ) -> tuple[list[str], int, str, dict]:
        base_path = shutil.which("gemini")
        cmd = [base_path, "-y", "-m", model, "-o", "stream-json"]
        if iteration == 0:
            cmd.extend(["-p", initial_prompt])
            stdin_text = None
        else:
            cmd.extend(["--resume", "latest"])
            stdin_text = feedback

        raw_lines = []
        num_turns = 0
        token_stats = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}

        proc = subprocess.Popen(
            cmd,
            cwd=str(ws_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,  # isolate process group so we can kill the whole tree
        )
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise ValueError("Failed to open stdin, stdout or stderr")
        if stdin_text:
            proc.stdin.write(stdin_text)
        proc.stdin.close()

        line_queue: queue.Queue = queue.Queue()

        def _reader():
            try:
                for line in proc.stdout or []:
                    line_queue.put(line)
            finally:
                line_queue.put(None)

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        session_start_time = time.time()
        session_timeout = 10800

        def _parse_event(line_str):
            nonlocal num_turns, token_stats
            obj = capture_raw_output_line(raw_lines, line_str)
            if obj is None:
                return
            evt_type = obj.get("type")
            if evt_type == "tool_use":
                num_turns += 1
            elif evt_type == "result":
                stats = obj.get("stats", {})
                token_stats = {
                    "input_tokens": stats.get("input", 0),
                    "cached_tokens": stats.get("cached", 0),
                    "output_tokens": stats.get("output_tokens", 0),
                }

        while True:
            remaining = session_timeout - (time.time() - session_start_time)
            if remaining <= 0:
                proc.terminate()
                break
            try:
                line = line_queue.get(timeout=min(remaining, 60))
            except queue.Empty:
                continue
            if line is None:
                break
            _parse_event(line)

        while True:
            try:
                line = line_queue.get(timeout=5)
            except queue.Empty:
                break
            if line is None:
                break
            _parse_event(line)

        stderr_text = ""
        try:
            proc.wait(timeout=30)
            stderr_text = proc.stderr.read()
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            with contextlib.suppress(OSError):
                stderr_text = proc.stderr.read()

        reader_thread.join(timeout=5)

        if num_turns == 0 and token_stats["input_tokens"] == 0:
            return raw_lines, num_turns, stderr_text, token_stats

        return raw_lines, num_turns, stderr_text, token_stats

    def extract_grid_from_output(self, raw_lines: list[str]) -> list[list[int]] | None:
        write_file_text = ""
        tool_result_text = ""

        for line in raw_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt_type = obj.get("type", "")
            if evt_type == "tool_use" and obj.get("tool_name") in (
                "write_file",
                "write_new_file",
            ):
                fpath = obj.get("parameters", {}).get("file_path", "").lower()
                if not fpath.endswith(".py") and any(
                    kw in fpath for kw in ("output", "answer", "result", "solution", "submission")
                ):
                    write_file_text += obj.get("parameters", {}).get("content", "") + "\n"
            elif evt_type == "tool_result":
                output = obj.get("output", "")
                if isinstance(output, str):
                    tool_result_text += output + "\n"

        grid = find_last_grid(write_file_text)
        if grid is not None:
            return grid
        return find_last_grid(tool_result_text)

    def _map_tool_params(self, gemini_name: str, params: dict[str, Any]) -> dict[str, Any]:
        if gemini_name == "run_shell_command":
            return {
                "command": params.get("command", ""),
                "description": params.get("description", ""),
            }
        if gemini_name == "read_file":
            return {"file_path": params.get("file_path", "")}
        if gemini_name in ("write_file", "write_new_file"):
            return {
                "file_path": params.get("file_path", ""),
                "content": params.get("content", ""),
            }
        if gemini_name == "edit_file":
            return {
                "file_path": params.get("file_path", ""),
                "old_string": params.get("old_string", ""),
                "new_string": params.get("new_string", ""),
                "replace_all": params.get("replace_all", False),
            }
        if gemini_name == "glob":
            return {"pattern": params.get("pattern", "")}
        if gemini_name == "grep":
            return {
                "pattern": params.get("pattern", ""),
                "path": params.get("path", ""),
            }
        if gemini_name == "list_directory":
            return {"pattern": params.get("dir_path", "") + "/*"}
        return params

    def _extract_grid_from_submit_cmd(self, cmd: str) -> list[list[int]] | None:
        match = re.search(r"submit\.py\s+['\"]?(\[.+\])['\"]?\s*$", cmd)
        if match:
            try:
                grid = json.loads(match.group(1))
                if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                    return grid
            except json.JSONDecodeError:
                pass
        return None

    def write_readable_log(self, rf: TextIO, obj: dict[str, Any]):
        evt_type = obj.get("type", "")
        if evt_type == "message" and obj.get("role") == "assistant":
            content = obj.get("content", "")
            if obj.get("delta"):
                rf.write(content)
            else:
                rf.write(f"\n**Assistant:**\n{content}\n\n")
        elif evt_type == "tool_use":
            tool_name = obj.get("tool_name", "")
            params = obj.get("parameters", {})
            if tool_name == "run_shell_command":
                rf.write(f"\n\n**Tool: {tool_name}**\n```\n$ {params.get('command', '')}\n```\n\n")
            else:
                input_str = json.dumps(params, indent=2)[:500]
                rf.write(f"\n\n**Tool: {tool_name}**\n```\n{input_str}\n```\n\n")
        elif evt_type == "tool_result":
            output = obj.get("output", "")[:2000]
            status = obj.get("status", "")
            rf.write(f"**Tool Result ({status}):**\n```\n{output}\n```\n\n")
        elif evt_type == "result":
            stats = obj.get("stats", {})
            rf.write(
                f"---\n**Result:** tokens={stats.get('total_tokens', '?')}, duration={stats.get('duration_ms', '?')}ms, tool_calls={stats.get('tool_calls', '?')}\n"
            )

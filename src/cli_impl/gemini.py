import json
import queue
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

SOLVER_MD = """\
# ARC-AGI Puzzle Solver

Read `task.json`. It has `train` (input/output pairs) and `test` (test input(s)).
Find the transformation pattern and apply it to the test input(s).

Use `python3` for scripting. All common scientific/mathematical packages are pre-installed — use whatever you need.

Output grids must contain only integers 0-9.

## Approach
Write `transform.py` with a Python function `transform(grid: np.ndarray) -> np.ndarray`.
The function takes a 2D numpy integer array and returns a 2D numpy integer array.
Test against ALL training pairs. Iterate until correct.

When analyzing, consider: object manipulation, color changes, spatial patterns,
object relationships, grid structure (borders, separators, subgrids).
"""


def get_solver_md() -> str:
    return SOLVER_MD


def workspace_extras(ws_path: Path):
    gemini_dir = ws_path / ".gemini"
    gemini_dir.mkdir(parents=True, exist_ok=True)
    settings = json.dumps(
        {
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
    (gemini_dir / "settings.json").write_text(settings)


GEMINI_PRICING = {
    "gemini-3-flash-preview": (0.50, 3.00, 0.05),
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
    "gemini-2.5-pro": (1.25, 10.00, 0.125),
}


def calculate_cost(
    model: str, input_tokens: int, cached_tokens: int, output_tokens: int
) -> float:
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
    ws_path: Path,
    model: str,
    initial_prompt: str,
    feedback: str,
    iteration: int,
    session_started: bool,
    task_id: str,
    test_index: int,
    _status_cb,
) -> tuple[list[str], int, str, dict, bool]:
    gemini_cmd_base = f"gemini -y -m {model} -o stream-json"
    if not session_started:
        cmd_str = f"{gemini_cmd_base} -p {shlex.quote(initial_prompt)}"
        stdin_text = None
    else:
        cmd_str = f"{gemini_cmd_base} --resume latest"
        stdin_text = feedback

    full_cmd = f"cd {ws_path} && {cmd_str}"
    proc = subprocess.Popen(
        ["bash", "-c", full_cmd],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if stdin_text:
        proc.stdin.write(stdin_text)
    proc.stdin.close()

    line_queue: queue.Queue = queue.Queue()

    def _reader():
        try:
            for line in proc.stdout:
                line_queue.put(line)
        finally:
            line_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    raw_lines = []
    num_turns = 0
    token_stats = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    session_start_time = time.time()
    session_timeout = 10800

    def _parse_event(line_str):
        nonlocal num_turns, token_stats
        line_str = line_str.rstrip("\n").rstrip("\r")
        if not line_str:
            return
        raw_lines.append(line_str)
        try:
            obj = json.loads(line_str)
        except json.JSONDecodeError:
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
        try:
            stderr_text = proc.stderr.read()
        except Exception:
            pass

    reader_thread.join(timeout=5)

    if num_turns == 0 and token_stats["input_tokens"] == 0:
        return raw_lines, num_turns, stderr_text, token_stats, session_started

    return raw_lines, num_turns, stderr_text, token_stats, True


def _find_last_grid(text: str) -> Optional[list[list[int]]]:
    if not text:
        return None
    grids = []
    i = 0
    while i < len(text):
        if text[i] == "[" and i + 1 < len(text) and text[i + 1] in "[ \n\r\t":
            depth = 0
            j = i
            while j < len(text):
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        try:
                            parsed = json.loads(candidate)
                            if (
                                isinstance(parsed, list)
                                and len(parsed) > 0
                                and all(isinstance(row, list) for row in parsed)
                                and all(
                                    isinstance(v, int) and 0 <= v <= 9
                                    for row in parsed
                                    for v in row
                                )
                            ):
                                grids.append(parsed)
                        except (json.JSONDecodeError, TypeError):
                            pass
                        break
                j += 1
        i += 1
    return grids[-1] if grids else None


def extract_grid_from_output(raw_lines: list[str]) -> Optional[list[list[int]]]:
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
                kw in fpath
                for kw in ("output", "answer", "result", "solution", "submission")
            ):
                write_file_text += obj.get("parameters", {}).get("content", "") + "\n"
        elif evt_type == "tool_result":
            output = obj.get("output", "")
            if isinstance(output, str):
                tool_result_text += output + "\n"

    grid = _find_last_grid(write_file_text)
    if grid is not None:
        return grid
    return _find_last_grid(tool_result_text)


# Orchestrator parsing
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


def _map_tool_params(gemini_name: str, params: dict) -> dict:
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
        return {"pattern": params.get("pattern", ""), "path": params.get("path", "")}
    if gemini_name == "list_directory":
        return {"pattern": params.get("dir_path", "") + "/*"}
    return params


def extract_grid_from_submit_cmd(cmd: str) -> Optional[list[list[int]]]:
    match = re.search(r"submit\.py\s+['\"]?(\[.+\])['\"]?\s*$", cmd)
    if match:
        try:
            grid = json.loads(match.group(1))
            if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                return grid
        except json.JSONDecodeError:
            pass
    return None


def parse_stream_json(raw_lines: list[str], task_id: str) -> list[dict]:
    entries = []
    turn_counter = 0
    current_blocks = []
    pending_text = ""

    def flush_text():
        nonlocal pending_text
        if pending_text.strip():
            current_blocks.append({"type": "text", "text": pending_text.strip()})
        pending_text = ""

    def flush_assistant():
        nonlocal current_blocks, turn_counter
        if current_blocks:
            turn_counter += 1
            entries.append(
                {"type": "assistant", "turn": turn_counter, "content": current_blocks}
            )
            current_blocks = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        evt_type = obj.get("type", "")

        if evt_type == "message":
            role = obj.get("role", "")
            content = obj.get("content", "")
            is_delta = obj.get("delta", False)
            if role == "assistant":
                if is_delta:
                    pending_text += content
                else:
                    flush_text()
                    if content.strip():
                        current_blocks.append({"type": "text", "text": content.strip()})

        elif evt_type == "tool_use":
            flush_text()
            gemini_name = obj.get("tool_name", "")
            tool_id = obj.get("tool_id", "")
            params = obj.get("parameters", {})

            viewer_name = _TOOL_NAME_MAP.get(gemini_name, gemini_name)
            viewer_params = _map_tool_params(gemini_name, params)

            if gemini_name == "run_shell_command" and "submit.py" in params.get(
                "command", ""
            ):
                cmd = params.get("command", "")
                grid = extract_grid_from_submit_cmd(cmd)
                if grid is not None:
                    current_blocks.append(
                        {
                            "type": "tool_use",
                            "name": "submit",
                            "id": tool_id,
                            "input": {"output": grid, "test_index": 0},
                        }
                    )
                else:
                    current_blocks.append(
                        {
                            "type": "tool_use",
                            "name": viewer_name,
                            "id": tool_id,
                            "input": viewer_params,
                        }
                    )
            else:
                current_blocks.append(
                    {
                        "type": "tool_use",
                        "name": viewer_name,
                        "id": tool_id,
                        "input": viewer_params,
                    }
                )

        elif evt_type == "tool_result":
            flush_text()
            flush_assistant()
            tool_id = obj.get("tool_id", "")
            status = obj.get("status", "")
            output = obj.get("output", "")
            is_error = status == "error"
            if isinstance(output, str) and len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            entries.append(
                {
                    "type": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": output,
                            **({"is_error": True} if is_error else {}),
                        }
                    ],
                }
            )

        elif evt_type == "result":
            flush_text()
            flush_assistant()
            stats = obj.get("stats", {})
            entries.append(
                {
                    "type": "result",
                    "cost": 0,
                    "num_turns": turn_counter,
                    "usage": {
                        "input_tokens": stats.get("input_tokens", 0),
                        "output_tokens": stats.get("output_tokens", 0),
                        "total_tokens": stats.get("total_tokens", 0),
                        "cached_tokens": stats.get("cached", 0),
                    },
                }
            )

    flush_text()
    flush_assistant()
    return entries


def write_readable_log(rf, line: str, obj: dict):
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

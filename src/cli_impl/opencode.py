import json
import subprocess
from pathlib import Path
from typing import Optional, List, Any

from .base import CLIImpl

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

_TOOL_NAME_MAP = {
    "bash": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "list": "Glob",
    "task": "Task",
}

class OpenCodeCLI(CLIImpl):
    def get_solver_md(self) -> str:
        return SOLVER_MD

    def setup_workspace(
        self, ws_path: Path, raw_task: dict, test_index: int, seed: int = 0, whole_task: bool = False
    ):
        pass

    def workspace_extras(self, ws_path: Path):
        pass

    def calculate_cost(
        self, model: str, input_tokens: int, cached_tokens: int, output_tokens: int
    ) -> float:
        return 0.0

    def run_session(
        self,
        ws_path: Path,
        model: str,
        initial_prompt: str,
        feedback: str,
        iteration: int,
        session_started: bool,
        task_id: str,
        test_index: int,
        _status_cb: Any,
    ) -> tuple[List[str], int, str, dict, bool]:
        # Returns (raw_lines, turns, stderr, stats, session_started_now)
        cmd_args = ["run", "--format", "json"]
        if iteration == 0:
            cmd_args.extend(
                ["--model", model, "--title", f"ARC-{task_id}-t{test_index}", initial_prompt]
            )
        else:
            cmd_args.extend(["--continue", feedback])

        full_cmd = ["opencode"] + cmd_args
        proc = subprocess.Popen(
            full_cmd,
            cwd=str(ws_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise ValueError("Failed to open stdin, stdout or stderr")
        proc.stdin.close()
        raw_lines = []
        num_turns = 0
        token_stats = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}

        for line in proc.stdout or []:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            raw_lines.append(line)
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            evt_type = obj.get("type", "")
            if evt_type == "step_start":
                num_turns += 1
            elif evt_type == "step_finish":
                part = obj.get("part", {})
                tokens = part.get("tokens", {})
                token_stats["input_tokens"] += tokens.get("input", 0)
                token_stats["cached_tokens"] += tokens.get("cache", {}).get("read", 0)
                token_stats["output_tokens"] += tokens.get("output", 0)

        stderr_text = proc.stderr.read()
        proc.wait(timeout=3500)

        return raw_lines, num_turns, stderr_text, token_stats, True

    def extract_grid_from_output(self, raw_lines: List[str]) -> Optional[List[List[int]]]:
        all_text = ""
        for line in raw_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt_type = obj.get("type", "")
            if evt_type == "tool_use":
                part = obj.get("part", {})
                tool = part.get("tool", "")
                state = part.get("state", {})
                if tool.lower() in ("write", "edit"):
                    inp = state.get("input", {})
                    fpath = inp.get("filePath", inp.get("file_path", "")).lower()
                    if not fpath.endswith(".py") and any(
                        kw in fpath
                        for kw in ("output", "answer", "result", "solution", "submission")
                    ):
                        content = inp.get("content", "")
                        all_text += content + "\n"
                elif tool.lower() == "bash":
                    output = state.get("output", "")
                    if output:
                        all_text += output + "\n"
            elif evt_type == "text":
                part = obj.get("part", {})
                text = part.get("text", "")
                if text:
                    all_text += text + "\n"
        return self._find_last_grid(all_text)

    def _find_last_grid(self, text: str) -> Optional[List[List[int]]]:
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

    def _map_tool_params(self, tool_name: str, params: dict) -> dict:
        tool_lower = tool_name.lower()
        if tool_lower == "bash":
            return {
                "command": params.get("command", ""),
                "description": params.get("description", ""),
            }
        if tool_lower == "read":
            return {"file_path": params.get("file_path", "")}
        if tool_lower in ("write", "edit"):
            return {
                "file_path": params.get("file_path", ""),
                "content": params.get("content", ""),
            }
        if tool_lower == "glob":
            return {"pattern": params.get("pattern", "")}
        if tool_lower == "grep":
            return {"pattern": params.get("pattern", ""), "path": params.get("path", "")}
        return params

    def parse_stream_json(self, raw_lines: List[str], task_id: str) -> List[dict]:
        entries = []
        turn_counter = 0
        current_blocks = []
        total_tokens = {"input": 0, "output": 0, "cache_read": 0}

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
            part = obj.get("part", {})

            if evt_type == "text":
                text = part.get("text", "")
                if text.strip():
                    current_blocks.append({"type": "text", "text": text.strip()})

            elif evt_type == "tool_use":
                tool_name = part.get("tool", "")
                call_id = part.get("callID", "")
                state = part.get("state", {})
                inp = state.get("input", {})
                output = state.get("output", "")

                viewer_name = _TOOL_NAME_MAP.get(tool_name.lower(), tool_name)
                viewer_params = self._map_tool_params(tool_name, inp)

                current_blocks.append(
                    {
                        "type": "tool_use",
                        "name": viewer_name,
                        "id": call_id,
                        "input": viewer_params,
                    }
                )

                if output and len(output) > 10:
                    flush_assistant()
                    truncated = output[:5000] if len(output) > 5000 else output
                    entries.append(
                        {
                            "type": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": call_id,
                                    "content": truncated,
                                }
                            ],
                        }
                    )

            elif evt_type == "step_finish":
                flush_assistant()
                tokens = part.get("tokens", {})
                total_tokens["input"] += tokens.get("input", 0)
                total_tokens["output"] += tokens.get("output", 0)
                total_tokens["cache_read"] += tokens.get("cache", {}).get("read", 0)

        flush_assistant()

        if total_tokens["input"] > 0 or total_tokens["output"] > 0:
            entries.append(
                {
                    "type": "result",
                    "cost": 0,
                    "num_turns": turn_counter,
                    "usage": {
                        "input_tokens": total_tokens["input"],
                        "output_tokens": total_tokens["output"],
                        "total_tokens": total_tokens["input"] + total_tokens["output"],
                        "cached_tokens": total_tokens["cache_read"],
                    },
                }
            )

        return entries

    def write_readable_log(self, rf: Any, line: str, obj: dict):
        evt_type = obj.get("type", "")
        part = obj.get("part", {})
        if evt_type == "text":
            text = part.get("text", "")
            if text.strip():
                rf.write(f"\n**Assistant:**\n{text.strip()}\n\n")
        elif evt_type == "tool_use":
            tool_name = part.get("tool", "")
            state = part.get("state", {})
            inp = state.get("input", {})
            output = state.get("output", "")
            if tool_name.lower() == "bash":
                rf.write(
                    f"\n\n**Tool: {tool_name}**\n```\n$ {inp.get('command', '')}\n```\n\n"
                )
            else:
                input_str = json.dumps(inp, indent=2)[:500]
                rf.write(f"\n\n**Tool: {tool_name}**\n```\n{input_str}\n```\n\n")
            if output:
                truncated = output[:2000] if len(output) > 2000 else output
                rf.write(f"**Tool Result:**\n```\n{truncated}\n```\n\n")
        elif evt_type == "step_finish":
            tokens = part.get("tokens", {})
            rf.write(
                f"---\n**Step:** tokens={tokens.get('input', 0) + tokens.get('output', 0)}, reason={part.get('reason', '?')}\n"
            )

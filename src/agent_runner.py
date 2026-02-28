#!/usr/bin/env python3
"""ARC-AGI OpenCode CLI Agent Runner — runs inside an isolated Docker container.

Self-contained script that runs OpenCode CLI sessions to solve an ARC task.

Communication:
  - Config:  Reads /root/config.json (then DELETES it)
  - Status:  Prints JSON lines to stdout (orchestrator reads via on_stdout)
  - Results: Writes /workspace/results.json
"""

import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def emit_status(event: dict) -> None:
    """Print a JSON status event to stdout for the orchestrator."""
    try:
        print(json.dumps(event), flush=True)
    except Exception:
        pass


AGENTS_MD = """\
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


def extract_grid_from_output(raw_lines: list[str]) -> list[list[int]] | None:
    """Extract the agent's final answer grid from the OpenCode stream output.

    OpenCode JSON format:
    - type: "tool_use" with part.tool = "write"/"edit", part.state.input.content
    - type: "tool_use" with part.tool = "bash", part.state.output contains command output
    """
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

    return _find_last_grid(all_text)


def _find_last_grid(text: str) -> list[list[int]] | None:
    """Find the last valid 2D integer grid in text."""
    if not text:
        return None

    grids: list[list[list[int]]] = []
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


def test_transform(
    transform_path: Path, train_examples: list[dict]
) -> tuple[bool, str, "callable | None"]:
    """Load transform.py from disk, test against training examples.

    Returns (all_pass, feedback_text, fn_or_None).
    """
    try:
        spec = importlib.util.spec_from_file_location("transform", str(transform_path))
        if spec is None or spec.loader is None:
            return False, f"Could not load {transform_path}", None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        return False, f"Import error in transform.py:\n{traceback.format_exc()}", None

    if not hasattr(mod, "transform"):
        return False, "transform.py has no function named 'transform'", None

    fn = mod.transform

    for i, ex in enumerate(train_examples):
        inp = np.array(ex["input"], dtype=int)
        expected = np.array(ex["output"], dtype=int)

        try:
            result = fn(inp.copy())
        except Exception:
            return (
                False,
                "Your transform function doesn't pass the training examples. "
                "Try a fundamentally different approach.",
                None,
            )

        if not isinstance(result, np.ndarray):
            return (
                False,
                "Your transform function doesn't pass the training examples. "
                "Try a fundamentally different approach.",
                None,
            )

        result = result.astype(int)

        if not np.array_equal(result, expected):
            return (
                False,
                "Your transform function doesn't pass the training examples. "
                "Try a fundamentally different approach.",
                None,
            )

    return True, "All training examples pass.", fn


def prepare_workspace(
    agent_id: str,
    raw_task: dict,
    test_index: int,
) -> Path:
    """Create workspace at /workspace with task.json and AGENTS.md.

    Returns the workspace Path.
    """
    ws = Path("/workspace")
    ws.mkdir(parents=True, exist_ok=True)

    public_task = {
        "train": raw_task["train"],
        "test": [{"input": raw_task["test"][test_index]["input"]}],
    }
    (ws / "task.json").write_text(json.dumps(public_task, indent=2))

    (ws / "AGENTS.md").write_text(AGENTS_MD)

    return ws


def _run_opencode_session(
    cmd_args: list[str],
    cwd: str,
    stdin_text: str | None = None,
    session_timeout: float = 3500,
) -> tuple[list[str], int, str, dict[str, int]]:
    """Launch OpenCode CLI as a subprocess, read JSON output, wait for exit.

    Args:
        cmd_args: List of command arguments (e.g. ["run", "--format", "json", "..."])
        cwd: Working directory.
        stdin_text: If provided, write to stdin then close.
        session_timeout: Max seconds for the entire session.

    Returns:
        (raw_lines, num_turns, stderr_text, token_stats)
    """
    full_cmd = ["opencode"] + cmd_args
    proc = subprocess.Popen(
        full_cmd,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if stdin_text:
        proc.stdin.write(stdin_text)
    proc.stdin.close()

    raw_lines: list[str] = []
    num_turns = 0
    token_stats: dict[str, int] = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
    }

    for line in proc.stdout:
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
    proc.wait(timeout=session_timeout)

    return raw_lines, num_turns, stderr_text, token_stats


def run_agent(config: dict) -> dict:
    """Run OpenCode CLI agent to solve an ARC task."""
    task_id: str = config["task_id"]
    agent_id: str = config["agent_id"]
    raw_task: dict = config["raw_task"]
    test_index: int = config["test_index"]
    model: str = config["model"]
    max_iterations: int = config["max_iterations"]
    soft_training_feedback: bool = config["soft_training_feedback"]

    start = time.time()
    attempts: list[dict] = []
    attempts_used = 0
    total_turns = 0
    total_input_tokens = 0
    total_cached_tokens = 0
    total_output_tokens = 0
    stderr_text = ""
    all_raw_lines: list[str] = []

    def _status(event: dict) -> None:
        emit_status({"agent_id": agent_id, "task_id": task_id, **event})

    try:
        ws = prepare_workspace(agent_id, raw_task, test_index)

        _status({"event": "started", "model": model})

        initial_prompt = "Read AGENTS.md, then solve the ARC puzzle in task.json."

        feedback = ""
        last_outcome = "pending"

        for iteration in range(max_iterations):
            _status(
                {
                    "event": "iteration",
                    "iteration": iteration + 1,
                    "max_iterations": max_iterations,
                }
            )

            if iteration == 0:
                cmd_args = [
                    "run",
                    "--model",
                    model,
                    "--format",
                    "json",
                    "--title",
                    f"ARC-{task_id}-t{test_index}",
                    initial_prompt,
                ]
                raw_lines, turns, stderr, stats = _run_opencode_session(
                    cmd_args,
                    str(ws),
                    stdin_text=None,
                )
            else:
                cmd_args = [
                    "run",
                    "--continue",
                    "--format",
                    "json",
                    feedback,
                ]
                raw_lines, turns, stderr, stats = _run_opencode_session(
                    cmd_args,
                    str(ws),
                    stdin_text=None,
                )

            all_raw_lines.extend(raw_lines)
            total_turns += turns
            total_input_tokens += stats["input_tokens"]
            total_cached_tokens += stats["cached_tokens"]
            total_output_tokens += stats["output_tokens"]
            if stderr:
                _status({"event": "error", "msg": stderr})
                stderr_text += stderr + "\n"

            transform_path = ws / "transform.py"
            if not transform_path.exists():
                feedback = (
                    "You haven't written transform.py yet. Write a file called "
                    "transform.py with a function transform(grid: np.ndarray) -> np.ndarray."
                )
                last_outcome = "no_transform_py"
            else:
                all_pass, feedback_text, fn = test_transform(
                    transform_path,
                    raw_task["train"],
                )

                _status(
                    {
                        "event": "transform_validation",
                        "iteration": iteration + 1,
                        "all_pass": all_pass,
                    }
                )

                if not all_pass:
                    if soft_training_feedback:
                        feedback = (
                            "Your transform function doesn't pass the training "
                            "examples. Try again."
                        )
                    else:
                        feedback = feedback_text
                    last_outcome = "training_fail"
                elif fn is not None:
                    test_input = raw_task["test"][test_index]["input"]
                    try:
                        test_arr = np.array(test_input, dtype=int)
                        grid = fn(test_arr.copy()).astype(int).tolist()
                    except Exception as e:
                        feedback = (
                            f"Transform passed training but failed on test input: {e}"
                        )
                        last_outcome = "test_error"
                    else:
                        attempts_used += 1
                        attempts.append(
                            {
                                "test_index": test_index,
                                "attempt": attempts_used,
                                "grid": grid,
                                "timestamp": time.time(),
                            }
                        )
                        last_outcome = "submitted"
                        _status({"event": "submitted", "attempt": attempts_used})
                        break

        if not attempts:
            extracted = extract_grid_from_output(all_raw_lines)
            if extracted is not None:
                attempts_used += 1
                attempts.append(
                    {
                        "test_index": test_index,
                        "attempt": attempts_used,
                        "grid": extracted,
                        "timestamp": time.time(),
                    }
                )

        elapsed = time.time() - start

        _status(
            {
                "event": "done",
                "elapsed": round(elapsed, 1),
                "attempts": attempts_used,
            }
        )

        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "test_index": test_index,
            "attempts": attempts,
            "elapsed": round(elapsed, 1),
            "cost": 0.0,
            "turns": total_turns,
            "usage": {
                "input_tokens": total_input_tokens,
                "cached_tokens": total_cached_tokens,
                "output_tokens": total_output_tokens,
            },
            "timed_out": False,
            "raw_lines": all_raw_lines,
            "stderr": stderr_text,
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "test_index": test_index,
            "attempts": attempts,
            "elapsed": round(elapsed, 1),
            "cost": 0.0,
            "turns": total_turns,
            "usage": {
                "input_tokens": total_input_tokens,
                "cached_tokens": total_cached_tokens,
                "output_tokens": total_output_tokens,
            },
            "error": str(e),
            "raw_lines": all_raw_lines,
        }


def main() -> None:
    """Read config, delete it, run agent, write results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    config_path = Path("/root/config.json")
    if not config_path.exists():
        emit_status({"event": "error", "msg": "No /root/config.json found"})
        sys.exit(1)

    auth_path = Path("/root/.local/share/opencode")
    auth_path.mkdir(parents=True, exist_ok=True)
    with open(auth_path / "auth.json", "w") as f:
        json.dump(
            {
                "github-copilot": {
                    "type": "oauth",
                    "access": "",
                    "refresh": os.environ.get("GITHUB_TOKEN"),
                    "expires": 0,
                }
            },
            f,
        )

    config = json.loads(config_path.read_text())

    config_path.unlink()

    result = run_agent(config)

    results_path = Path("/workspace/results.json")
    results_path.write_text(json.dumps(result))

    emit_status({"event": "results_written", "path": str(results_path)})


if __name__ == "__main__":
    main()

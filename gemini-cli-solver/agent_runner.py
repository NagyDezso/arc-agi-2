#!/usr/bin/env python3
"""ARC-AGI Gemini CLI Agent Runner — runs inside E2B sandbox.

Self-contained script that runs Gemini CLI sessions to solve an ARC task.

Communication:
  - Config:  Reads /root/config.json (then DELETES it)
  - Status:  Prints JSON lines to stdout (orchestrator reads via on_stdout)
  - Results: Writes /workspace/results.json
"""

import importlib.util
import json
import multiprocessing
import os
import random
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np


# ── Status Output ──────────────────────────────────────────────────────────

def emit_status(event: dict) -> None:
    """Print a JSON status event to stdout for the orchestrator."""
    try:
        print(json.dumps(event), flush=True)
    except Exception:
        pass


# ── GEMINI.md prompt ───────────────────────────────────────────────────────

GEMINI_MD = """\
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

# ── Gemini API Pricing ────────────────────────────────────────────────────────

# (input, output, cached) per 1M tokens — <=200K tier for tiered models
GEMINI_PRICING: dict[str, tuple[float, float, float]] = {
    "gemini-3-flash-preview":  (0.50,  3.00,  0.05),
    "gemini-2.5-flash":        (0.30,  2.50,  0.03),
    "gemini-3.1-pro-preview":  (2.00, 12.00,  0.20),
    "gemini-2.5-pro":          (1.25, 10.00,  0.125),
}


def calculate_cost(
    model: str,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate USD cost from token counts using the pricing table."""
    pricing = GEMINI_PRICING.get(model)
    if pricing is None:
        return 0.0
    input_rate, output_rate, cached_rate = pricing
    return (
        input_tokens * input_rate / 1_000_000
        + cached_tokens * cached_rate / 1_000_000
        + output_tokens * output_rate / 1_000_000
    )


# ── Grid extraction ─────────────────────────────────────────────────────────

def extract_grid_from_output(raw_lines: list[str]) -> list[list[int]] | None:
    """Extract the agent's final answer grid from the stream output.

    Strategy (based on observed Gemini CLI behavior):
    1. If the agent wrote to an output/answer/result file, use the last grid
       from that content (most intentional signal).
    2. Otherwise, use the last grid from tool_result events — this is where
       Python script stdout appears (e.g. print(json.dumps(grid))).

    We deliberately ignore assistant text messages: Gemini often recites the
    grid from memory in its summary, introducing transcription errors. The
    authoritative source is always the actual program output in tool_result.
    """
    write_file_text = ""
    tool_result_text = ""

    for line in raw_lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        evt_type = obj.get("type", "")

        if evt_type == "tool_use" and obj.get("tool_name") in ("write_file", "write_new_file"):
            fpath = obj.get("parameters", {}).get("file_path", "").lower()
            if not fpath.endswith(".py") and any(
                kw in fpath for kw in ("output", "answer", "result", "solution", "submission")
            ):
                write_file_text += obj.get("parameters", {}).get("content", "") + "\n"

        elif evt_type == "tool_result":
            output = obj.get("output", "")
            if isinstance(output, str):
                tool_result_text += output + "\n"

    # Prefer explicit write_file to output/answer file
    grid = _find_last_grid(write_file_text)
    if grid is not None:
        return grid

    # Fall back to last grid in tool_result output
    return _find_last_grid(tool_result_text)


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


# ── Timeout wrapper for transform execution ──────────────────────────────────

TRANSFORM_TIMEOUT = 120  # seconds — kill transforms that take too long (infinite loops)


def _run_fn_in_proc(fn, arg, result_queue):
    """Target for subprocess: run fn(arg) and put result on queue."""
    try:
        result = fn(arg)
        result_queue.put(("ok", result))
    except Exception as e:
        result_queue.put(("error", e))


def run_with_timeout(fn, arg, timeout=TRANSFORM_TIMEOUT):
    """Run fn(arg) with a timeout. Returns result or raises TimeoutError/Exception."""
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_run_fn_in_proc, args=(fn, arg, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        raise TimeoutError(f"Transform execution timed out after {timeout}s (likely infinite loop)")
    if q.empty():
        raise RuntimeError("Transform process died without returning a result")
    status, value = q.get_nowait()
    if status == "error":
        raise value
    return value


# ── Transform validation ─────────────────────────────────────────────────────

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
            result = run_with_timeout(fn, inp.copy())
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


# ── Workspace setup (local filesystem) ────────────────────────────────────

def prepare_workspace(
    agent_id: str,
    raw_task: dict,
    test_index: int,
    seed: int = 0,
    whole_task: bool = False,
) -> Path:
    """Create workspace at /workspace with task.json, GEMINI.md, and settings.

    Returns the workspace Path.
    """
    ws = Path("/workspace")
    ws.mkdir(parents=True, exist_ok=True)
    gemini_dir = ws / ".gemini"
    gemini_dir.mkdir(parents=True, exist_ok=True)

    # Optionally shuffle training examples (seeded by ensemble index)
    import random
    train = list(raw_task["train"])
    if len(train) > 1 and seed > 0:
        rng = random.Random(seed)
        rng.shuffle(train)

    # task.json: train + test input(s) (no output/answer)
    if whole_task:
        public_task = {
            "train": train,
            "test": [{"input": t["input"]} for t in raw_task["test"]],
        }
    else:
        public_task = {
            "train": train,
            "test": [{"input": raw_task["test"][test_index]["input"]}],
        }
    (ws / "task.json").write_text(json.dumps(public_task, indent=2))

    (ws / "GEMINI.md").write_text(GEMINI_MD)

    # .gemini/settings.json — let agents run as long as they need
    settings = json.dumps({
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
    }, indent=2)
    (ws / ".gemini" / "settings.json").write_text(settings)

    return ws


# ── Gemini CLI session helper (subprocess-based) ─────────────────────────

def _run_gemini_session(
    cmd_str: str,
    cwd: str,
    stdin_text: str | None = None,
    session_timeout: float = 10800,  # 3 hours default
) -> tuple[list[str], int, str, dict[str, int]]:
    """Launch gemini CLI as a subprocess, read stream-json output, wait for exit.

    Uses a reader thread + queue so we can enforce a session timeout.
    The process is allowed to run until it exits naturally or hits the
    session timeout.

    Args:
        cmd_str: Shell command string to run (gemini invocation).
        cwd: Working directory.
        stdin_text: If provided, write to stdin then close (for --resume mode).
        session_timeout: Max seconds for the entire session (default: 3h).

    Returns:
        (raw_lines, num_turns, stderr_text, token_stats)
        where token_stats has keys: input_tokens, cached_tokens, output_tokens
    """
    import queue
    import threading

    full_cmd = f"cd {cwd} && {cmd_str}"
    proc = subprocess.Popen(
        ["bash", "-c", full_cmd],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # If resuming, pipe the feedback prompt then close stdin;
    # otherwise close stdin immediately so the process doesn't hang.
    if stdin_text:
        proc.stdin.write(stdin_text)
    proc.stdin.close()

    # Reader thread: pushes lines (or None for EOF) into a queue
    line_queue: queue.Queue[str | None] = queue.Queue()

    def _reader():
        try:
            for line in proc.stdout:
                line_queue.put(line)
        finally:
            line_queue.put(None)  # sentinel for EOF

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    raw_lines: list[str] = []
    num_turns = 0
    token_stats: dict[str, int] = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    session_start = time.time()

    def _parse_event(line_str):
        """Parse a stream-json line and update turns/token_stats."""
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
        remaining = session_timeout - (time.time() - session_start)
        if remaining <= 0:
            proc.terminate()  # SIGTERM: give CLI a chance to emit final stats
            break

        try:
            line = line_queue.get(timeout=min(remaining, 60))
        except queue.Empty:
            continue  # Check session timeout again

        if line is None:  # EOF — process closed stdout
            break

        _parse_event(line)

    # Drain remaining queue items — CLI may emit a final "result" event
    # with token stats during SIGTERM graceful shutdown
    while True:
        try:
            line = line_queue.get(timeout=5)
        except queue.Empty:
            break
        if line is None:
            break
        _parse_event(line)

    # Wait for process to finish and capture stderr
    stderr_text = ""
    try:
        proc.wait(timeout=30)
        stderr_text = proc.stderr.read()
    except subprocess.TimeoutExpired:
        proc.kill()  # SIGKILL as last resort if SIGTERM didn't work
        proc.wait()
        try:
            stderr_text = proc.stderr.read()
        except Exception:
            pass

    reader_thread.join(timeout=5)

    return raw_lines, num_turns, stderr_text, token_stats


# ── Main agent logic ─────────────────────────────────────────────────────

def run_agent(config: dict) -> dict:
    """Run Gemini CLI agent to solve an ARC task."""
    task_id: str = config["task_id"]
    agent_id: str = config["agent_id"]
    raw_task: dict = config["raw_task"]
    test_index: int = config["test_index"]
    model: str = config["model"]
    max_iterations: int = config.get("max_iterations", 10)
    soft_training_feedback: bool = config.get("soft_training_feedback", False)
    whole_task: bool = config.get("whole_task", False)

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
        # Extract ensemble index from agent_id (e.g. "taskid_ens5" -> 5)
        import re
        ens_match = re.search(r'_ens(\d+)', agent_id)
        seed = int(ens_match.group(1)) if ens_match else 0

        ws = prepare_workspace(agent_id, raw_task, test_index, seed=seed, whole_task=whole_task)

        _status({"event": "started", "model": model})

        initial_prompt = "Read GEMINI.md, then solve the ARC puzzle in task.json."
        gemini_cmd_base = f"gemini -y -m {model} -o stream-json"

        # ── Transform loop: validate + retry with --resume ──
        feedback = ""
        last_outcome = "pending"

        iteration = 0
        empty_retries = 0
        session_started = False  # Track if first successful session has happened
        while iteration < max_iterations:
            _status({"event": "iteration", "iteration": iteration + 1, "max_iterations": max_iterations})

            if not session_started:
                cmd_str = f"{gemini_cmd_base} -p {shlex.quote(initial_prompt)}"
                raw_lines, turns, stderr, stats = _run_gemini_session(
                    cmd_str, str(ws), stdin_text=None,
                )
            else:
                cmd_str = f"{gemini_cmd_base} --resume latest"
                raw_lines, turns, stderr, stats = _run_gemini_session(
                    cmd_str, str(ws), stdin_text=feedback,
                )

            all_raw_lines.extend(raw_lines)
            total_turns += turns
            total_input_tokens += stats["input_tokens"]
            total_cached_tokens += stats["cached_tokens"]
            total_output_tokens += stats["output_tokens"]
            if stderr:
                stderr_text += stderr + "\n"

            # If session produced nothing (API error / network failure), retry
            # with exponential backoff + jitter to survive sustained API outages
            if turns == 0 and stats["input_tokens"] == 0:
                empty_retries += 1
                _status({"event": "empty_session", "iteration": iteration + 1, "retry": empty_retries})
                if empty_retries >= 12:
                    _status({"event": "too_many_empty_sessions"})
                    break
                base_delay = min(30 * (2 ** (empty_retries - 1)), 300)  # 30s, 60s, 120s, 240s, 300s, 300s...
                jitter = random.uniform(0, base_delay * 0.3)
                delay = base_delay + jitter
                _status({"event": "empty_session_backoff", "delay": round(delay, 1)})
                time.sleep(delay)
                continue

            session_started = True

            # Read transform.py from workspace (Gemini wrote it via write_file)
            transform_path = ws / "transform.py"
            if not transform_path.exists():
                feedback = (
                    "You haven't written transform.py yet. Write a file called "
                    "transform.py with a function transform(grid: np.ndarray) -> np.ndarray."
                )
                last_outcome = "no_transform_py"
            else:
                # Validate transform against training examples
                all_pass, feedback_text, fn = test_transform(
                    transform_path, raw_task["train"],
                )

                _status({
                    "event": "transform_validation",
                    "iteration": iteration + 1,
                    "all_pass": all_pass,
                })

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
                    if whole_task:
                        # Apply transform to ALL test inputs
                        all_ok = True
                        pending_grids = []
                        for ti, test_case in enumerate(raw_task["test"]):
                            try:
                                test_arr = np.array(test_case["input"], dtype=int)
                                grid = run_with_timeout(fn, test_arr.copy()).astype(int).tolist()
                            except Exception as e:
                                feedback = f"Transform passed training but failed on test input {ti}: {e}"
                                last_outcome = "test_error"
                                all_ok = False
                                break
                            pending_grids.append({"test_index": ti, "grid": grid})
                        if all_ok:
                            for p in pending_grids:
                                attempts_used += 1
                                attempts.append({
                                    **p,
                                    "attempt": attempts_used,
                                    "timestamp": time.time(),
                                })
                            last_outcome = "submitted"
                            _status({"event": "submitted", "attempt": attempts_used})
                            break
                    else:
                        test_input = raw_task["test"][test_index]["input"]
                        try:
                            test_arr = np.array(test_input, dtype=int)
                            grid = run_with_timeout(fn, test_arr.copy()).astype(int).tolist()
                        except Exception as e:
                            feedback = f"Transform passed training but failed on test input: {e}"
                            last_outcome = "test_error"
                        else:
                            attempts_used += 1
                            attempts.append({
                                "test_index": test_index,
                                "attempt": attempts_used,
                                "grid": grid,
                                "timestamp": time.time(),
                            })
                            last_outcome = "submitted"
                            _status({"event": "submitted", "attempt": attempts_used})
                            # Training passed — submit the grid and stop
                            break

            iteration += 1

        # Fallback: heuristic extraction if transform loop produced nothing
        # (only for single-test mode; can't produce grids for multiple test inputs)
        if not attempts and not whole_task:
            extracted = extract_grid_from_output(all_raw_lines)
            if extracted is not None:
                attempts_used += 1
                attempts.append({
                    "test_index": test_index,
                    "attempt": attempts_used,
                    "grid": extracted,
                    "timestamp": time.time(),
                })

        elapsed = time.time() - start

        _status({
            "event": "done",
            "elapsed": round(elapsed, 1),
            "attempts": attempts_used,
        })

        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "test_index": test_index,
            "attempts": attempts,
            "elapsed": round(elapsed, 1),
            "cost": calculate_cost(model, total_input_tokens, total_cached_tokens, total_output_tokens),
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
            "cost": calculate_cost(model, total_input_tokens, total_cached_tokens, total_output_tokens),
            "turns": total_turns,
            "usage": {
                "input_tokens": total_input_tokens,
                "cached_tokens": total_cached_tokens,
                "output_tokens": total_output_tokens,
            },
            "error": str(e),
            "raw_lines": all_raw_lines,
        }


# ── Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    """Read config, delete it, run agent, write results."""
    config_path = Path("/root/config.json")
    if not config_path.exists():
        emit_status({"event": "error", "msg": "No /root/config.json found"})
        sys.exit(1)

    config = json.loads(config_path.read_text())

    # DELETE config before starting Gemini CLI — removes any possibility
    # of the agent reading orchestrator config.
    config_path.unlink()

    result = run_agent(config)

    results_path = Path("/workspace/results.json")
    results_path.write_text(json.dumps(result))

    emit_status({"event": "results_written", "path": str(results_path)})


if __name__ == "__main__":
    main()

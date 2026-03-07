#!/usr/bin/env python3
"""ARC-AGI Generic Agent Runner — runs inside isolated container/sandbox.

Self-contained script that runs CLI sessions to solve an ARC task.
"""

import importlib.util
import json
import logging
import multiprocessing
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from cli_impl import CLIImpl


logger = logging.getLogger(__name__)

INSTRUCTION = """\
Read `task.json`. It has `train` (input/output pairs) and `test` (one test input).
Find the transformation pattern and apply it to the test input.
Use `python3` for scripting. All common scientific/mathematical packages are pre-installed — use whatever you need.
Output grids must contain only integers 0-9.
Write `transform.py` with a Python function `transform(grid: np.ndarray) -> np.ndarray`.
The function takes a 2D numpy integer array and returns a 2D numpy integer array.
Test against ALL training pairs. Iterate until correct.
When analyzing, consider: object manipulation, color changes, spatial patterns,
object relationships, grid structure (borders, separators, subgrids).
"""


def emit_status(event: dict) -> None:
    try:
        print(json.dumps(event), flush=True)
    except Exception:
        pass


TRANSFORM_TIMEOUT = 120


def _run_fn_in_proc(fn, arg, result_queue):
    try:
        result = fn(arg)
        result_queue.put(("ok", result))
    except Exception as e:
        result_queue.put(("error", e))


def run_with_timeout(fn, arg, timeout=TRANSFORM_TIMEOUT):
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


def _format_diff(expected: np.ndarray, actual: np.ndarray) -> str:
    """Format expected vs actual diff for feedback."""
    lines = []
    if expected.shape != actual.shape:
        lines.append(f"Shape mismatch: expected {expected.shape}, got {actual.shape}")
    else:
        diff_mask = expected != actual
        n_diff = int(np.sum(diff_mask))
        lines.append(f"Value mismatch: {n_diff} cell(s) differ")
        if n_diff > 0:
            diff_where = np.argwhere(diff_mask)
            shown = min(len(diff_where), 10)
            lines.append("First differing cells (row, col): expected -> actual:")
            for idx in range(shown):
                r, c = diff_where[idx]
                lines.append(f"  ({r},{c}): {expected[r, c]} -> {actual[r, c]}")
            if n_diff > shown:
                lines.append(f"  ... and {n_diff - shown} more")
    lines.append("")
    lines.append("Expected output:")
    lines.append(np.array2string(expected, max_line_width=120, threshold=100))
    lines.append("")
    lines.append("Your output:")
    lines.append(np.array2string(actual, max_line_width=120, threshold=100))
    return "\n".join(lines)


def test_transform(transform_path: Path, train_examples: list[dict]) -> tuple[bool, str, Callable | None]:
    try:
        spec = importlib.util.spec_from_file_location("transform", str(transform_path))
        if spec is None or spec.loader is None:
            return False, f"Could not load {transform_path}", None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        return False, f"Import error in transform.py:\n{traceback.format_exc()}", None

    if not hasattr(mod, "transform"):
        return False, "transform.py has no function named 'transform'", None
    fn = mod.transform

    for i, ex in enumerate(train_examples):
        inp = np.array(ex["input"], dtype=int)
        expected = np.array(ex["output"], dtype=int)
        try:
            result = run_with_timeout(fn, inp.copy())
        except Exception as e:
            return (
                False,
                f"Training example {i} (0-indexed): transform raised {type(e).__name__}: {e}\n"
                "Try a fundamentally different approach.",
                None,
            )
        if not isinstance(result, np.ndarray):
            return (
                False,
                f"Training example {i} (0-indexed): transform returned {type(result).__name__}, expected np.ndarray.\n"
                "Try a fundamentally different approach.",
                None,
            )
        result = result.astype(int)
        if not np.array_equal(result, expected):
            diff = _format_diff(expected, result)
            return (
                False,
                f"Training example {i} (0-indexed) failed.\n{diff}\nFix the transform to match expected output.",
                None,
            )

    return True, "All training examples pass.", fn


def prepare_workspace(
    agent_id: str,
    raw_task: dict,
    test_index: int,
    cli_impl: CLIImpl,
    seed: int = 0,
    whole_task: bool = False,
) -> Path:
    ws = Path("/workspace")
    ws.mkdir(parents=True, exist_ok=True)
    train = list(raw_task["train"])
    if len(train) > 1 and seed > 0:
        rng = random.Random(seed)
        rng.shuffle(train)

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

    cli_impl.workspace_extras(ws)
    return ws


def run_agent(config: dict) -> dict:
    task_id: str = config["task_id"]
    agent_id: str = config["agent_id"]
    raw_task: dict = config["raw_task"]
    test_index: int = config["test_index"]
    model: str = config["model"]
    max_iterations: int = config.get("max_iterations", 10)
    soft_training_feedback: bool = config.get("soft_training_feedback", False)
    whole_task: bool = config.get("whole_task", False)
    cli_type: str = config.get("cli_type", "gemini")

    import cli_impl

    impl = cli_impl.get_cli_impl(cli_type)

    start = time.time()
    attempts, attempts_used = [], 0
    total_turns, total_input_tokens, total_cached_tokens, total_output_tokens = (
        0,
        0,
        0,
        0,
    )
    stderr_text = ""
    all_raw_lines = []

    def _status(event: dict) -> None:
        emit_status({"agent_id": agent_id, "task_id": task_id, **event})

    try:
        import re

        ens_match = re.search(r"_ens(\d+)", agent_id)
        seed = int(ens_match.group(1)) if ens_match else 0
        ws = prepare_workspace(agent_id, raw_task, test_index, impl, seed=seed, whole_task=whole_task)
        _status({"event": "started", "model": model})
        feedback = ""
        iteration = 0
        session_started = False
        while iteration < max_iterations:
            _status(
                {
                    "event": "iteration",
                    "iteration": iteration + 1,
                    "max_iterations": max_iterations,
                }
            )

            raw_lines, turns, stderr, stats, session_started = impl.run_session(
                ws_path=ws,
                model=model,
                initial_prompt=INSTRUCTION,
                feedback=feedback,
                iteration=iteration,
                session_started=session_started,
                task_id=task_id,
                test_index=test_index,
                _status_cb=_status,
            )

            all_raw_lines.extend(raw_lines)
            total_turns += turns
            total_input_tokens += stats["input_tokens"]
            total_cached_tokens += stats["cached_tokens"]
            total_output_tokens += stats["output_tokens"]
            if stderr:
                _status({"event": "error", "msg": stderr})
                stderr_text += stderr + "\n"
                # Check for fatal errors that should stop the agent
                fatal_errors = [
                    "ModelNotFoundError",
                    "Invalid model",
                    "Requested entity was not found",
                    "The model is not supported",
                    "Access denied",
                    "API key not valid",
                    "QuotaExceeded",
                ]
                if any(err in stderr for err in fatal_errors):
                    break

            transform_path = ws / "transform.py"
            if not transform_path.exists():
                feedback = (
                    "You haven't written transform.py yet. Write a file called "
                    "transform.py with a function transform(grid: np.ndarray) -> np.ndarray."
                )
            else:
                all_pass, feedback_text, fn = test_transform(transform_path, raw_task["train"])
                _status(
                    {
                        "event": "transform_validation",
                        "iteration": iteration + 1,
                        "all_pass": all_pass,
                    }
                )

                if not all_pass:
                    feedback = (
                        "Your transform function doesn't pass the training examples. Try again."
                        if soft_training_feedback
                        else feedback_text
                    )
                elif fn is not None:
                    if whole_task:
                        all_ok = True
                        pending_grids = []
                        for ti, test_case in enumerate(raw_task["test"]):
                            try:
                                test_arr = np.array(test_case["input"], dtype=int)
                                grid = run_with_timeout(fn, test_arr.copy()).astype(int).tolist()
                            except Exception as e:
                                feedback = f"Transform passed training but failed on test input {ti}: {e}"
                                all_ok = False
                                break
                            pending_grids.append({"test_index": ti, "grid": grid})
                        if all_ok:
                            for p in pending_grids:
                                attempts_used += 1
                                attempts.append(
                                    {
                                        **p,
                                        "attempt": attempts_used,
                                        "timestamp": time.time(),
                                    }
                                )
                            _status({"event": "submitted", "attempt": attempts_used})
                            break
                    else:
                        test_input = raw_task["test"][test_index]["input"]
                        try:
                            test_arr = np.array(test_input, dtype=int)
                            grid = run_with_timeout(fn, test_arr.copy()).astype(int).tolist()
                        except Exception as e:
                            feedback = f"Transform passed training but failed on test input: {e}"
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
                            _status({"event": "submitted", "attempt": attempts_used})
                            break

            iteration += 1

        if not attempts and not whole_task:
            extracted = impl.extract_grid_from_output(all_raw_lines)
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
        cost = impl.calculate_cost(model, total_input_tokens, total_cached_tokens, total_output_tokens)
        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "test_index": test_index,
            "attempts": attempts,
            "elapsed": round(elapsed, 1),
            "cost": cost,
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
        cost = impl.calculate_cost(model, total_input_tokens, total_cached_tokens, total_output_tokens)
        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "test_index": test_index,
            "attempts": attempts,
            "elapsed": round(elapsed, 1),
            "cost": cost,
            "turns": total_turns,
            "usage": {
                "input_tokens": total_input_tokens,
                "cached_tokens": total_cached_tokens,
                "output_tokens": total_output_tokens,
            },
            "error": str(e),
            "raw_lines": all_raw_lines,
        }


def main():
    config_path = Path("/root/config.json")
    if not config_path.exists():
        emit_status({"event": "error", "msg": "No /root/config.json found"})
        sys.exit(1)

    # OpenCode auth initialization
    auth_path = Path("/root/.local/share/opencode")
    auth_path.mkdir(parents=True, exist_ok=True)
    with open(auth_path / "auth.json", "w") as f:
        json.dump(
            {
                "github-copilot": {
                    "type": "oauth",
                    "access": "",
                    "refresh": os.environ.get("GITHUB_TOKEN", ""),
                    "expires": 0,
                }
            },
            f,
        )

    # Gemini OAuth initialization
    gemini_access_token = os.environ.get("GEMINI_OAUTH_ACCESS_TOKEN")
    if gemini_access_token:
        gemini_auth_path = Path("/root/.gemini")
        gemini_auth_path.mkdir(parents=True, exist_ok=True)

        (gemini_auth_path / "oauth_creds.json").write_text(
            json.dumps(
                {
                    "access_token": gemini_access_token,
                    "refresh_token": os.environ.get("GEMINI_OAUTH_REFRESH_TOKEN"),
                    "scope": "https://www.googleapis.com/auth/userinfo.email openid https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/cloud-platform",
                    "token_type": "Bearer",
                    "id_token": os.environ.get("GEMINI_OAUTH_ID_TOKEN"),
                    "expiry_date": 1772303384460,
                }
            )
        )
        (gemini_auth_path / "settings.json").write_text(
            json.dumps({"security": {"auth": {"selectedType": "oauth-personal"}}})
        )

    config = json.loads(config_path.read_text())
    config_path.unlink()

    result = run_agent(config)
    results_path = Path("/workspace/results.json")
    results_path.write_text(json.dumps(result))
    emit_status({"event": "results_written", "path": str(results_path)})


if __name__ == "__main__":
    main()

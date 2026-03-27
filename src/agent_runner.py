#!/usr/bin/env python3
"""ARC-AGI Generic Agent Runner — runs inside isolated container/sandbox.

Self-contained script that runs CLI sessions to solve an ARC task.
"""

import importlib.util
import json
import multiprocessing
import random
import re
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

from cli_impl import BaseCLI, Event, EventType, get_cli_impl
from models import AgentAttempt, AgentConfig, AgentResultData, UsageTotals

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

FATAL_ERRORS = [
    "ModelNotFoundError",
    "Invalid model",
    "Requested entity was not found",
    "The model is not supported",
    "Access denied",
    "API key not valid",
    "QuotaExceeded",
]


TRANSFORM_TIMEOUT = 120


def _run_fn_in_proc(fn, arg, result_queue):
    try:
        result = fn(arg)
        result_queue.put(("ok", result))
    except Exception as error:
        result_queue.put(("error", error))


def run_with_timeout(fn, arg, timeout=TRANSFORM_TIMEOUT):
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_run_fn_in_proc, args=(fn, arg, result_queue))
    process.start()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join()
        raise TimeoutError(f"Transform execution timed out after {timeout}s (likely infinite loop)")
    if result_queue.empty():
        raise RuntimeError("Transform process died without returning a result")
    status, value = result_queue.get_nowait()
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
                row, col = diff_where[idx]
                lines.append(f"  ({row},{col}): {expected[row, col]} -> {actual[row, col]}")
            if n_diff > shown:
                lines.append(f"  ... and {n_diff - shown} more")
    lines.append("")
    lines.append("Expected output:")
    lines.append(np.array2string(expected, max_line_width=120, threshold=100))
    lines.append("")
    lines.append("Your output:")
    lines.append(np.array2string(actual, max_line_width=120, threshold=100))
    return "\n".join(lines)


def run_transform(transform_path: Path, train_examples: list[dict]) -> tuple[bool, str, Callable | None]:
    try:
        spec = importlib.util.spec_from_file_location("transform", str(transform_path))
        if spec is None or spec.loader is None:
            return False, f"Could not load {transform_path}", None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return False, f"Import error in transform.py:\n{traceback.format_exc()}", None

    if not hasattr(module, "transform"):
        return False, "transform.py has no function named 'transform'", None
    fn = module.transform

    for index, example in enumerate(train_examples):
        inp = np.array(example["input"], dtype=int)
        expected = np.array(example["output"], dtype=int)
        try:
            result = run_with_timeout(fn, inp.copy())
        except Exception as error:
            return (
                False,
                f"Training example {index} (0-indexed): transform raised {type(error).__name__}: {error}\n"
                "Try a fundamentally different approach.",
                None,
            )
        if not isinstance(result, np.ndarray):
            return (
                False,
                f"Training example {index} (0-indexed): transform returned {type(result).__name__}, expected np.ndarray.\n"
                "Try a fundamentally different approach.",
                None,
            )
        result = result.astype(int)
        if not np.array_equal(result, expected):
            diff = _format_diff(expected, result)
            return (
                False,
                f"Training example {index} (0-indexed) failed.\n{diff}\nFix the transform to match expected output.",
                None,
            )

    return True, "All training examples pass.", fn


def prepare_workspace(
    raw_task: dict,
    test_index: int,
    cli_impl: BaseCLI,
    seed: int = 0,
    whole_task: bool = False,
) -> Path:
    ws_path = Path("/workspace")
    ws_path.mkdir(parents=True, exist_ok=True)
    train_examples = list(raw_task["train"])
    if len(train_examples) > 1 and seed > 0:
        rng = random.Random(seed)
        rng.shuffle(train_examples)

    if whole_task:
        public_task = {
            "train": train_examples,
            "test": [{"input": test_case["input"]} for test_case in raw_task["test"]],
        }
    else:
        public_task = {
            "train": train_examples,
            "test": [{"input": raw_task["test"][test_index]["input"]}],
        }
    (ws_path / "task.json").write_text(json.dumps(public_task))
    cli_impl.workspace_extras(ws_path)
    return ws_path


def _emit_status(agent_id: str, message: str, *, level: str = "info") -> None:
    print(Event(type=EventType.STATUS, message=f"[{agent_id}] {message}", level=level).model_dump_json(), flush=True)


def _emit_harness_feedback(all_raw_lines: list[str], for_iteration: int, text: str) -> None:
    line = json.dumps({"type": "harness_feedback", "for_iteration": for_iteration, "text": text})
    all_raw_lines.append(line)
    print(Event(type=EventType.TRANSCRIPT, message=line).model_dump_json(), flush=True)


def run_agent(config: AgentConfig, cli: BaseCLI) -> AgentResultData:
    start_time = time.time()
    attempts: list[AgentAttempt] = []
    attempts_used = 0
    total_turns = 0
    total_input_tokens = 0
    total_cached_tokens = 0
    total_output_tokens = 0
    stderr_text = ""
    all_raw_lines: list[str] = []

    try:
        ens_match = re.search(r"_ens(\d+)", config.agent_id)
        seed = int(ens_match.group(1)) if ens_match else 0
        ws_path = prepare_workspace(config.raw_task, config.test_index, cli, seed=seed, whole_task=config.whole_task)

        _emit_status(config.agent_id, f"started model={config.model}")
        feedback = ""
        iteration = 0

        while iteration < config.max_iterations:
            _emit_status(config.agent_id, f"iteration {iteration + 1}/{config.max_iterations}")
            if feedback:
                _emit_harness_feedback(all_raw_lines, iteration + 1, feedback)
            raw_lines, turns, stderr, stats = cli.run_session(
                ws_path=ws_path, model=config.model, initial_prompt=INSTRUCTION, feedback=feedback, iteration=iteration
            )

            all_raw_lines.extend(raw_lines)
            total_turns += turns
            total_input_tokens += stats["input_tokens"]
            total_cached_tokens += stats["cached_tokens"]
            total_output_tokens += stats["output_tokens"]
            if stderr:
                _emit_status(config.agent_id, f"Error: {stderr.strip()}", level="error")
                stderr_text += stderr + "\n"
                # Check for fatal errors that should stop the agent
                if any(err in stderr for err in FATAL_ERRORS):
                    break

            transform_path = ws_path / "transform.py"
            if not transform_path.exists():
                feedback = (
                    "You haven't written transform.py yet. Write a file called "
                    "transform.py with a function transform(grid: np.ndarray) -> np.ndarray."
                )
            else:
                all_pass, feedback_text, fn = run_transform(transform_path, config.raw_task["train"])
                _emit_status(config.agent_id, f"validation iteration={iteration + 1} all_pass={all_pass}")

                if not all_pass:
                    feedback = (
                        "Your transform function doesn't pass the training examples. Try again."
                        if config.soft_training_feedback
                        else feedback_text
                    )
                elif fn is not None:
                    if config.whole_task:
                        all_ok = True
                        pending_grids: list[tuple[int, list[list[int]]]] = []
                        for candidate_index, test_case in enumerate(config.raw_task["test"]):
                            try:
                                test_arr = np.array(test_case["input"], dtype=int)
                                grid = run_with_timeout(fn, test_arr.copy()).astype(int).tolist()
                            except Exception as error:
                                feedback = (
                                    f"Transform passed training but failed on test input {candidate_index}: {error}"
                                )
                                all_ok = False
                                break
                            pending_grids.append((candidate_index, grid))
                        if all_ok:
                            for candidate_index, grid in pending_grids:
                                attempts_used += 1
                                attempts.append(
                                    AgentAttempt(
                                        task_id=config.task_id,
                                        attempt=attempts_used,
                                        test_index=candidate_index,
                                        grid=grid,
                                    )
                                )
                            _emit_status(config.agent_id, f"submitted attempt={attempts_used}")
                            break
                    else:
                        test_input = config.raw_task["test"][config.test_index]["input"]
                        try:
                            test_arr = np.array(test_input, dtype=int)
                            grid = run_with_timeout(fn, test_arr.copy()).astype(int).tolist()
                        except Exception as error:
                            feedback = f"Transform passed training but failed on test input: {error}"
                        else:
                            attempts_used += 1
                            attempts.append(
                                AgentAttempt(
                                    task_id=config.task_id,
                                    attempt=attempts_used,
                                    test_index=config.test_index,
                                    grid=grid,
                                )
                            )
                            _emit_status(config.agent_id, f"submitted attempt={attempts_used}")
                            break
            iteration += 1

        if not attempts and not config.whole_task:
            extracted = cli.extract_grid_from_output(all_raw_lines)
            if extracted is not None:
                attempts_used += 1
                attempts.append(
                    AgentAttempt(
                        task_id=config.task_id,
                        attempt=attempts_used,
                        test_index=config.test_index,
                        grid=extracted,
                    )
                )

        elapsed = time.time() - start_time
        _emit_status(config.agent_id, f"done elapsed={round(elapsed, 1)}s attempts={attempts_used}")
        cost = cli.calculate_cost(config.model, total_input_tokens, total_cached_tokens, total_output_tokens)
        return AgentResultData(
            task_id=config.task_id,
            agent_id=config.agent_id,
            test_index=config.test_index,
            attempts=attempts,
            cost=cost,
            turns=total_turns,
            usage=UsageTotals(
                input_tokens=total_input_tokens, cached_tokens=total_cached_tokens, output_tokens=total_output_tokens
            ),
            elapsed=elapsed,
            raw_lines=all_raw_lines,
            stderr=stderr_text,
        )

    except Exception as error:
        elapsed = time.time() - start_time
        cost = cli.calculate_cost(config.model, total_input_tokens, total_cached_tokens, total_output_tokens)
        return AgentResultData(
            task_id=config.task_id,
            agent_id=config.agent_id,
            test_index=config.test_index,
            attempts=attempts,
            elapsed=elapsed,
            cost=cost,
            turns=total_turns,
            usage=UsageTotals(
                input_tokens=total_input_tokens,
                cached_tokens=total_cached_tokens,
                output_tokens=total_output_tokens,
            ),
            raw_lines=all_raw_lines,
            stderr=stderr_text,
            error=str(error),
        )


def main() -> None:
    config_path = Path("/root/config.json")
    if not config_path.exists():
        _emit_status("None", "startup error: missing /root/config.json", level="error")
        return

    config = AgentConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    config_path.unlink()
    cli = get_cli_impl(config.cli_type)
    result = run_agent(config, cli)
    results_path = Path("/workspace/results.json")
    results_path.write_text(result.model_dump_json(), encoding="utf-8")
    _emit_status(config.agent_id, f"results written path={results_path}")


if __name__ == "__main__":
    main()

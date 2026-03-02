"""Orchestrator: dispatches CLI agents to local Docker containers or E2B sandboxes.

The orchestrator runs locally and handles:
- Task loading from ARC-AGI data files
- Dispatching agents to backends
- Writing logs (raw_stream, transcript, readable, attempts)
- Results aggregation and summary.json
- Resume logic (skip completed tasks)
"""

import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
import random
from collections.abc import Callable
from pathlib import Path

from src.cli_impl import get_cli_impl, CLIImpl
from src.backends import get_backend_runner
from src.utils.logging import setup_logging

ROOT = Path(__file__).resolve().parent
CHALLENGES_FILE = ROOT.parent / "data" / "arc-agi_evaluation_challenges.json"
RESULTS = ROOT / "results"

_EVENT_FORMATTERS: dict[str, Callable[[dict], str]] = {
    "started": lambda e: f"started (model={e.get('model', '?')})",
    "iteration": lambda e: (
        f"iteration {e.get('iteration', '?')}/{e.get('max_iterations', '?')}"
    ),
    "transform_validation": lambda e: (
        f"transform {'PASS' if e.get('all_pass') else 'FAIL'} (iter {e.get('iteration', '?')})"
    ),
    "submitted": lambda e: f"submit #{e.get('attempt', '?')}",
    "done": lambda e: (
        f"done — {e.get('attempts', 0)} attempts, {e.get('elapsed', '?')}s"
    ),
    "results_written": lambda e: "results written",
    "error": lambda e: f"ERROR: {e.get('msg', '')}",
}

logger = logging.getLogger(__name__)

_ALL_TASKS: dict[str, dict] = {}


def _load_all_tasks() -> dict[str, dict]:
    global _ALL_TASKS
    if not _ALL_TASKS:
        if not CHALLENGES_FILE.exists():
            raise FileNotFoundError(f"Challenges file not found: {CHALLENGES_FILE}")
        _ALL_TASKS = json.loads(CHALLENGES_FILE.read_text())
    return _ALL_TASKS


def load_task_ids(tasks_arg: str) -> list[str]:
    if tasks_arg == "all":
        return sorted(_load_all_tasks().keys())
    return [t.strip() for t in tasks_arg.split(",") if t.strip()]


def load_task_json(task_id: str) -> dict:
    all_tasks = _load_all_tasks()
    if task_id not in all_tasks:
        raise KeyError(f"Task {task_id} not found")
    return all_tasks[task_id]


def write_agent_logs(result: dict, task_id: str, log_dir: Path, cli_impl: CLIImpl) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_lines: list[str] = result.get("raw_lines", [])

    # raw_stream.jsonl might have been written live by the backend
    raw_stream_path = log_dir / "raw_stream.jsonl"
    if not raw_stream_path.exists() and raw_lines:
        with open(raw_stream_path, "w") as f:
            for line in raw_lines:
                f.write(line + "\n")

    transcript_entries = cli_impl.parse_stream_json(raw_lines, task_id)
    with open(log_dir / "transcript.jsonl", "w") as f:
        for entry in transcript_entries:
            f.write(json.dumps(entry) + "\n")

    with open(log_dir / "readable.md", "w") as rf:
        agent_id = result.get("agent_id", "unknown")
        test_index = result.get("test_index", 0)
        rf.write(f"# Agent Log: {agent_id} (task: {task_id}, test: {test_index})\n\n")
        for line in raw_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                rf.write(f"[raw] {line}\n")
                continue
            cli_impl.write_readable_log(rf, line, obj)

    with open(log_dir / "attempts.jsonl", "w") as f:
        for attempt in result.get("attempts", []):
            f.write(json.dumps(attempt) + "\n")

    stderr = result.get("stderr", "")
    if stderr:
        (log_dir / "stderr.log").write_text(stderr)
    if "error" in result:
        (log_dir / "error.log").write_text(result["error"])


MAX_RETRIES = 10
INITIAL_BACKOFF_S = 2.0
MAX_BACKOFF_S = 600.0


async def _retry_backend_call(coro_fn, *, agent_id: str) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await coro_fn()
        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(
                kw in err_str
                for kw in (
                    "deadline exceeded",
                    "unavailable",
                    "connection",
                    "timeout",
                    "reset by peer",
                    "broken pipe",
                    "eof",
                    "transport",
                    "503",
                    "502",
                    "429",
                    "rate limit",
                    "resource_exhausted",
                    "overloaded",
                    "too many requests",
                    "stopped or disabled",
                )
            )
            if not is_transient or attempt == MAX_RETRIES:
                raise
            backoff = min(INITIAL_BACKOFF_S * (2 ** (attempt - 1)), MAX_BACKOFF_S)
            jitter = random.uniform(0, backoff * 0.5)
            wait = backoff + jitter
            logger.warning(
                f"[{agent_id}] Attempt {attempt}/{MAX_RETRIES} failed: {e} — retrying in {wait:.1f}s"
            )
            await asyncio.sleep(wait)
    raise RuntimeError(f"[{agent_id}] All {MAX_RETRIES} retries exhausted")


def _write_agent_result(
    run_dir: Path, task_id: str, agent_id: str, agent_data: dict
) -> None:
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    task_file = task_results_dir / f"{task_id}.json"
    tmp_file = task_results_dir / f"{task_id}.json.tmp"
    if task_file.exists():
        try:
            data = json.loads(task_file.read_text())
        except (json.JSONDecodeError, OSError):
            data = {"agents": {}}
    else:
        data = {"agents": {}}
    data.setdefault("agents", {})[agent_id] = agent_data
    tmp_file.write_text(json.dumps(data, indent=2))
    os.rename(str(tmp_file), str(task_file))


async def process_task(
    task_id: str,
    args: argparse.Namespace,
    run_dir: Path,
    backend_queue: asyncio.Queue | None,
    backend_impl,
    cli_impl,
) -> dict:
    raw_task = load_task_json(task_id)
    num_tests = len(raw_task["test"])
    agent_metas = []

    async def _dispatch(
        agent_id: str, kwargs: dict, test_index: int, log_dir: Path
    ) -> dict:
        max_empty_retries = 3

        async def _run_with_empty_retry():
            for empty_attempt in range(max_empty_retries + 1):
                result = await _retry_backend_call(
                    lambda kw={**kwargs, "log_dir": log_dir}: backend_impl.run_agent(**kw), 
                    agent_id=agent_id
                )
                turns = result.get("turns", 0)
                attempts = result.get("attempts", [])
                error = result.get("error")
                if turns > 0 or len(attempts) > 0 or error or empty_attempt >= max_empty_retries:
                    if (
                        turns == 0
                        and len(attempts) == 0
                        and not error
                        and empty_attempt >= max_empty_retries
                    ):
                        logger.warning(
                            f"  [empty] {agent_id}: all {max_empty_retries} sandbox "
                            f"retries exhausted, accepting empty result"
                        )
                    return result
                wait = 10 * (empty_attempt + 1)
                logger.info(
                    f"  [empty] {agent_id}: 0 turns/attempts, retrying sandbox "
                    f"({empty_attempt + 1}/{max_empty_retries}) in {wait}s..."
                )
                await asyncio.sleep(wait)
            return result

        if backend_queue is None:
            result = await _run_with_empty_retry()
        else:
            token = await backend_queue.get()
            try:
                result = await _run_with_empty_retry()
            finally:
                backend_queue.put_nowait(token)

        if not isinstance(result, BaseException):
            write_agent_logs(result, task_id, log_dir, cli_impl)
            attempts = result.get("attempts", [])
            agent_data = {
                "test_index": test_index,
                "attempts": [a["grid"] for a in attempts],
                "cost": result.get("cost", 0),
                "backend_cost": result.get("backend_cost", 0),
                "backend_duration": result.get("backend_duration", 0),
                "total_cost": result.get("total_cost", 0),
                "turns": result.get("turns", 0),
                "usage": result.get("usage", {}),
            }
            _write_agent_result(run_dir, task_id, agent_id, agent_data)
        return result

    agent_coros = []
    whole_task = getattr(args, "whole_task", False)
    if whole_task:
        for ei in range(args.num_agents):
            agent_id = f"{task_id}_ens{ei}"
            agent_log_dir = run_dir / "logs" / task_id / f"agent{ei}"
            agent_metas.append((agent_id, 0, agent_log_dir))
            _kwargs = dict(
                task_id=task_id,
                agent_id=agent_id,
                raw_task=raw_task,
                test_index=0,
                model=args.model,
                max_iterations=args.max_iterations,
                soft_training_feedback=args.soft_training_feedback,
                whole_task=True,
                cli_type=args.cli,
                root_path=ROOT,
            )
            agent_coros.append(_dispatch(agent_id, _kwargs, 0, agent_log_dir))
    else:
        for ti in range(num_tests):
            for ei in range(args.num_agents):
                agent_id = f"{task_id}_ens{ei}_t{ti}"
                agent_log_dir = run_dir / "logs" / task_id / f"t{ti}" / f"agent{ei}"
                agent_metas.append((agent_id, ti, agent_log_dir))
                _kwargs = dict(
                    task_id=task_id,
                    agent_id=agent_id,
                    raw_task=raw_task,
                    test_index=ti,
                    model=args.model,
                    max_iterations=args.max_iterations,
                    soft_training_feedback=args.soft_training_feedback,
                    whole_task=False,
                    cli_type=args.cli,
                    root_path=ROOT,
                )
                agent_coros.append(_dispatch(agent_id, _kwargs, ti, agent_log_dir))

    agent_results = await asyncio.gather(*agent_coros, return_exceptions=True)
    per_agent = {}
    submitted_tests = set()

    for (agent_id, ti, log_dir), result in zip(agent_metas, agent_results):
        if isinstance(result, BaseException):
            per_agent[agent_id] = {
                "test_index": ti,
                "attempts": [],
                "error": str(result),
            }
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "error.log").write_text(str(result))
            continue
        attempts = result.get("attempts", [])
        for a in attempts:
            if a.get("grid") is not None:
                submitted_tests.add(a.get("test_index", ti))
        per_agent[agent_id] = {
            "test_index": ti,
            "attempts": [a["grid"] for a in attempts],
            "cost": result.get("cost", 0),
            "backend_cost": result.get("backend_cost", 0),
            "backend_duration": result.get("backend_duration", 0),
            "total_cost": result.get("total_cost", 0),
            "turns": result.get("turns", 0),
            "usage": result.get("usage", {}),
        }

    submitted = len(submitted_tests)
    valid_results = [r for r in agent_results if isinstance(r, dict)]
    total_api_cost = sum(r.get("cost", 0) for r in valid_results)
    total_backend_cost = sum(r.get("backend_cost", 0) for r in valid_results)
    elapsed = max((r.get("elapsed", 0) for r in valid_results), default=0)

    total_usage = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    for r in valid_results:
        usage = r.get("usage", {})
        total_usage["input_tokens"] += usage.get("input_tokens", 0)
        total_usage["cached_tokens"] += usage.get("cached_tokens", 0)
        total_usage["output_tokens"] += usage.get("output_tokens", 0)

    score_data = {
        "submitted": submitted,
        "total": num_tests,
        "elapsed": round(elapsed, 1),
        "api_cost": round(total_api_cost, 4),
        "backend_cost": round(total_backend_cost, 4),
        "total_cost": round(total_api_cost + total_backend_cost, 4),
        "usage": total_usage,
    }
    task_result = {"score": score_data, "agents": per_agent}
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    (task_results_dir / f"{task_id}.json").write_text(json.dumps(task_result, indent=2))
    return {"task_id": task_id, "score": score_data}


async def run_all(args: argparse.Namespace):
    backend_impl = get_backend_runner(args.backend)
    cli = get_cli_impl(args.cli)
    try:
        await backend_impl.setup(ROOT, args.cli)
    except Exception as e:
        logger.error(f"Failed to setup backend: {e}")
        return

    task_ids = load_task_ids(args.tasks)

    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.is_absolute():
            run_dir = RESULTS / args.resume
        if not run_dir.exists():
            logger.error(f"Resume directory not found: {run_dir}")
            return
    else:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{args.name}_{run_stamp}" if args.name else run_stamp
        run_dir = RESULTS / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    logger.info(f"Loaded {len(task_ids)} tasks")

    RESULTS.mkdir(parents=True, exist_ok=True)
    latest = RESULTS / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)
    logger.info(f"Run directory: {run_dir}")

    completed_tasks = {}
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    for f in task_results_dir.glob("*.json"):
        try:
            completed_tasks[f.stem] = json.loads(f.read_text())
        except Exception:
            pass
    if completed_tasks:
        logger.info(f"Found {len(completed_tasks)} already-completed tasks, skipping them")

    remaining_ids = [tid for tid in task_ids if tid not in completed_tasks]
    if getattr(args, "limit", None):
        remaining_ids = remaining_ids[: args.limit]
    logger.info(
        f"Running {len(remaining_ids)} tasks ({len(task_ids) - len(remaining_ids)} skipped)"
    )

    all_scores = {}
    total_submitted, total_tests, total_cost = 0, 0, 0.0
    for tid, data in completed_tasks.items():
        score = data.get("score", {})
        all_scores[tid] = score
        total_submitted += score.get("submitted", 0)
        total_tests += score.get("total", 0)
        total_cost += score.get("total_cost", score.get("cost", 0))

    completed = len(completed_tasks)
    backend_queue = None
    if getattr(args, "concurrency", 0) > 0:
        backend_queue = asyncio.Queue()
        for _ in range(args.concurrency):
            backend_queue.put_nowait("slot")

    async def _process_and_report(task_id: str):
        nonlocal completed, total_submitted, total_tests, total_cost
        try:
            result = await process_task(
                task_id, args, run_dir, backend_queue, backend_impl, cli
            )
        except Exception as e:
            completed += 1
            logger.error(f"[{completed}/{len(task_ids)}] CRASH {task_id}: {e}", exc_info=True)
            return

        score = result["score"]
        total_submitted += score["submitted"]
        total_tests += score["total"]
        total_cost += score.get("total_cost", 0)
        all_scores[task_id] = score
        completed += 1
        s, t = score["submitted"], score["total"]
        logger.info(
            f"[{completed}/{len(task_ids)}] {'ok' if s == t else 'XX'} {task_id}  {s}/{t} submitted  ({score.get('elapsed', 0):.0f}s)"
        )

    random.shuffle(remaining_ids)
    await asyncio.gather(
        *[_process_and_report(tid) for tid in remaining_ids], return_exceptions=True
    )

    summary = {
        "cli": args.cli,
        "backend": args.backend,
        "model": args.model,
        "num_agents": args.num_agents,
        "max_iterations": args.max_iterations,
        "soft_training_feedback": args.soft_training_feedback,
        "whole_task": getattr(args, "whole_task", False),
        "num_tasks": len(task_ids),
        "total_tests": total_tests,
        "total_cost": round(total_cost, 2),
        "tasks": all_scores,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        f"\n{'=' * 60}\nDone! {len(task_ids)} tasks, {total_tests} test inputs\nSummary: {run_dir / 'summary.json'}"
    )

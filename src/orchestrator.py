"""Orchestrator: dispatches CLI agents to local Docker containers or E2B sandboxes.

The orchestrator runs locally and handles:
- Task loading from ARC-AGI data files
- Dispatching agents to backends
- Writing logs (session, transcript, readable, attempts)
- Results aggregation and summary.json
- Resume logic (skip completed tasks)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docker.errors import DockerException

from src.backends import get_backend_runner
from src.cli_impl import CLIImpl, get_cli_impl
from src.logger import setup_logging
from src.log_protocol import SESSION_LOG_FILENAME, TRANSCRIPT_FILENAME
from src.models import (
    AgentResultData,
    AgentRunSpec,
    OrchestrationContext,
    TaskProcessConfig,
    TaskProcessResult,
    TaskScore,
    UsageTotals,
)

if TYPE_CHECKING:
    import argparse
    from collections.abc import Awaitable, Callable, Mapping

ROOT = Path(__file__).resolve().parent
CHALLENGES_FILE = ROOT.parent / "data" / "arc-agi_evaluation_challenges.json"
RESULTS = ROOT / "results"

logger = logging.getLogger(__name__)

_ALL_TASKS: dict[str, dict[str, Any]] = {}

MAX_RETRIES = 10
INITIAL_BACKOFF_S = 2.0
MAX_BACKOFF_S = 600.0
MAX_EMPTY_RETRIES = 3


def _load_all_tasks() -> dict[str, dict[str, Any]]:
    if not _ALL_TASKS:
        if not CHALLENGES_FILE.exists():
            message = f"Challenges file not found: {CHALLENGES_FILE}"
            raise FileNotFoundError(message)
        _ALL_TASKS.update(json.loads(CHALLENGES_FILE.read_text(encoding="utf-8")))
    return _ALL_TASKS


def load_task_ids(tasks_arg: str) -> list[str]:
    if tasks_arg == "all":
        return sorted(_load_all_tasks().keys())
    return [task_id.strip() for task_id in tasks_arg.split(",") if task_id.strip()]


def load_task_json(task_id: str) -> dict[str, Any]:
    all_tasks = _load_all_tasks()
    if task_id not in all_tasks:
        message = f"Task {task_id} not found"
        raise KeyError(message)
    return all_tasks[task_id]


def write_agent_logs(
    result: dict[str, Any],
    task_id: str,
    log_dir: Path,
    cli_impl: CLIImpl,
    model: str | None = None,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_lines: list[str] = result.get("raw_lines", [])

    session_log_path = log_dir / SESSION_LOG_FILENAME
    if not session_log_path.exists():
        legacy_raw_stream = log_dir / "raw_stream.jsonl"
        if legacy_raw_stream.exists():
            legacy_raw_stream.replace(session_log_path)

    transcript_path = log_dir / TRANSCRIPT_FILENAME
    if not transcript_path.exists():
        transcript_entries = cli_impl.parse_stream_json(raw_lines, task_id, model=model)
        transcript_path.write_text(
            "".join(f"{json.dumps(entry)}\n" for entry in transcript_entries),
            encoding="utf-8",
            errors="replace",
        )

    readable_path = log_dir / "readable.md"
    with readable_path.open("w", encoding="utf-8", errors="replace") as readable_file:
        agent_id = result.get("agent_id", "unknown")
        test_index = result.get("test_index", 0)
        readable_file.write(f"# Agent Log: {agent_id} (task: {task_id}, test: {test_index})\n\n")
        for line in raw_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                readable_file.write(f"[raw] {line}\n")
                continue
            cli_impl.write_readable_log(readable_file, line, obj)

    attempts_path = log_dir / "attempts.jsonl"
    attempts_path.write_text(
        "".join(f"{json.dumps(attempt)}\n" for attempt in result.get("attempts", [])),
        encoding="utf-8",
        errors="replace",
    )

    stderr = result.get("stderr", "")
    if stderr:
        (log_dir / "stderr.log").write_text(stderr, encoding="utf-8", errors="replace")
    if "error" in result:
        (log_dir / "error.log").write_text(
            result["error"],
            encoding="utf-8",
            errors="replace",
        )


async def _retry_backend_call(
    coro_fn: Callable[[], Awaitable[dict[str, Any]]],
    *,
    agent_id: str,
) -> dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await coro_fn()
        except Exception as error:
            err_str = str(error).lower()
            is_transient = any(
                keyword in err_str
                for keyword in (
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
            logger.warning(f"[{agent_id}] Attempt {attempt}/{MAX_RETRIES} failed: {error} — retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
    message = f"[{agent_id}] All {MAX_RETRIES} retries exhausted"
    raise RuntimeError(message)


def _write_agent_result(
    run_dir: Path,
    task_id: str,
    agent_id: str,
    agent_data: dict[str, Any],
) -> None:
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    task_file = task_results_dir / f"{task_id}.json"
    tmp_file = task_results_dir / f"{task_id}.json.tmp"

    if task_file.exists():
        try:
            data = json.loads(task_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {"agents": {}}
    else:
        data = {"agents": {}}

    data.setdefault("agents", {})[agent_id] = agent_data
    tmp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_file.replace(task_file)


def get_envs(cli_type: str) -> dict[str, str]:
    envs: dict[str, str] = {}
    if cli_type == "opencode":
        kilo_key = os.environ.get("KILO_API_KEY")
        if kilo_key:
            envs["KILO_API_KEY"] = kilo_key
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            envs["GITHUB_TOKEN"] = github_token
    elif cli_type == "gemini":
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            envs["GEMINI_API_KEY"] = gemini_key
        gemini_oauth_access = os.environ.get("GEMINI_OAUTH_ACCESS_TOKEN")
        if gemini_oauth_access:
            envs["GEMINI_OAUTH_ACCESS_TOKEN"] = gemini_oauth_access
        gemini_oauth_refresh = os.environ.get("GEMINI_OAUTH_REFRESH_TOKEN")
        if gemini_oauth_refresh:
            envs["GEMINI_OAUTH_REFRESH_TOKEN"] = gemini_oauth_refresh
        gemini_oauth_id = os.environ.get("GEMINI_OAUTH_ID_TOKEN")
        if gemini_oauth_id:
            envs["GEMINI_OAUTH_ID_TOKEN"] = gemini_oauth_id
    return envs


def _build_agent_specs(
    task_id: str,
    raw_task: dict[str, Any],
    run_dir: Path,
    config: TaskProcessConfig,
) -> list[AgentRunSpec]:
    specs: list[AgentRunSpec] = []
    envs = get_envs(config.cli)
    test_indexes = [0] if config.whole_task else list(range(len(raw_task["test"])))
    for test_index in test_indexes:
        for ensemble_index in range(config.num_agents):
            if config.whole_task:
                agent_id = f"{task_id}_ens{ensemble_index}"
                log_dir = run_dir / "logs" / task_id / f"agent{ensemble_index}"
            else:
                agent_id = f"{task_id}_ens{ensemble_index}_t{test_index}"
                log_dir = run_dir / "logs" / task_id / f"t{test_index}" / f"agent{ensemble_index}"

            specs.append(
                AgentRunSpec(
                    task_id=task_id,
                    agent_id=agent_id,
                    test_index=test_index,
                    log_dir=log_dir,
                    raw_task=raw_task,
                    model=config.model,
                    envs=envs,
                    max_iterations=config.max_iterations,
                    soft_training_feedback=config.soft_training_feedback,
                    whole_task=config.whole_task,
                    cli_type=config.cli,
                    root_path=ROOT,
                )
            )

    return specs


async def _run_agent_with_empty_retry(spec: AgentRunSpec, context: OrchestrationContext) -> dict[str, Any]:
    result: dict[str, Any] | None = None

    for empty_attempt in range(MAX_EMPTY_RETRIES + 1):
        result = await _retry_backend_call(
            lambda: context.backend_impl.run_agent(spec),
            agent_id=spec.agent_id,
        )
        turns = int(result.get("turns", 0))
        attempts = result.get("attempts", [])
        error = result.get("error")

        if turns > 0 or len(attempts) > 0 or error or empty_attempt >= MAX_EMPTY_RETRIES:
            if turns == 0 and len(attempts) == 0 and not error and empty_attempt >= MAX_EMPTY_RETRIES:
                logger.warning(
                    f"  [empty] {spec.agent_id}: all {MAX_EMPTY_RETRIES} sandbox retries exhausted, "
                    "accepting empty result"
                )
            return result

        wait = 10 * (empty_attempt + 1)
        logger.info(
            f"  [empty] {spec.agent_id}: 0 turns/attempts, retrying sandbox "
            f"({empty_attempt + 1}/{MAX_EMPTY_RETRIES}) in {wait}s..."
        )
        await asyncio.sleep(wait)

    if result is None:
        message = f"No backend result available for {spec.agent_id}"
        raise RuntimeError(message)
    return result


async def _run_agent_spec(
    spec: AgentRunSpec,
    run_dir: Path,
    backend_queue: asyncio.Queue[Any] | None,
    context: OrchestrationContext,
) -> AgentResultData:
    if backend_queue is None:
        raw_result = await _run_agent_with_empty_retry(spec, context)
    else:
        token = await backend_queue.get()
        try:
            raw_result = await _run_agent_with_empty_retry(spec, context)
        finally:
            backend_queue.put_nowait(token)

    write_agent_logs(raw_result, spec.task_id, spec.log_dir, context.cli_impl, model=spec.model)

    agent_result = AgentResultData.from_backend_result(raw_result, spec.test_index)
    _write_agent_result(
        run_dir,
        spec.task_id,
        spec.agent_id,
        agent_result.to_persisted_agent_dict(),
    )
    return agent_result


def _merge_usage(results: list[AgentResultData]) -> UsageTotals:
    return UsageTotals(
        input_tokens=sum(result.usage.input_tokens for result in results),
        cached_tokens=sum(result.usage.cached_tokens for result in results),
        output_tokens=sum(result.usage.output_tokens for result in results),
    )


def _collect_task_result(
    task_id: str,
    num_tests: int,
    specs: list[AgentRunSpec],
    raw_results: list[AgentResultData | BaseException],
) -> tuple[dict[str, Any], TaskProcessResult]:
    per_agent: dict[str, Any] = {}
    submitted_tests: set[int] = set()
    valid_results: list[AgentResultData] = []

    for spec, raw_result in zip(specs, raw_results, strict=False):
        if isinstance(raw_result, BaseException):
            error_result = AgentResultData.from_exception(spec.agent_id, spec.test_index, raw_result)
            per_agent[spec.agent_id] = error_result.to_persisted_agent_dict()
            spec.log_dir.mkdir(parents=True, exist_ok=True)
            (spec.log_dir / "error.log").write_text(
                str(raw_result),
                encoding="utf-8",
                errors="replace",
            )
            continue

        valid_results.append(raw_result)
        per_agent[spec.agent_id] = raw_result.to_persisted_agent_dict()
        submitted_tests.update(raw_result.submitted_test_indexes())

    total_api_cost = sum(result.cost for result in valid_results)
    total_backend_cost = sum(result.backend_cost for result in valid_results)
    elapsed = max((result.elapsed for result in valid_results), default=0.0)
    usage = _merge_usage(valid_results)

    score = TaskScore(
        submitted=len(submitted_tests),
        total=num_tests,
        elapsed=round(elapsed, 1),
        api_cost=round(total_api_cost, 4),
        backend_cost=round(total_backend_cost, 4),
        total_cost=round(total_api_cost + total_backend_cost, 4),
        usage=usage,
    )
    task_result = {"score": score.to_dict(), "agents": per_agent}
    return task_result, TaskProcessResult(task_id=task_id, score=score)


async def process_task(
    task_id: str,
    args: argparse.Namespace | TaskProcessConfig,
    run_dir: Path,
    backend_queue: asyncio.Queue[Any] | None,
    context: OrchestrationContext,
) -> dict[str, Any]:
    config = args if isinstance(args, TaskProcessConfig) else TaskProcessConfig.from_args(args)
    raw_task = load_task_json(task_id)
    num_tests = len(raw_task["test"])
    specs = _build_agent_specs(task_id, raw_task, run_dir, config)

    agent_results = await asyncio.gather(
        *[_run_agent_spec(spec, run_dir, backend_queue, context) for spec in specs],
        return_exceptions=True,
    )

    task_result, process_result = _collect_task_result(task_id, num_tests, specs, agent_results)
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    (task_results_dir / f"{task_id}.json").write_text(
        json.dumps(task_result),
        encoding="utf-8",
    )
    return process_result.to_dict()


def _resolve_run_dir(args: argparse.Namespace) -> Path | None:
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.is_absolute():
            run_dir = RESULTS / args.resume
        if not run_dir.exists():
            logger.error(f"Resume directory not found: {run_dir}")
            return None
        return run_dir

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{args.name}_{run_stamp}" if args.name else run_stamp
    run_dir = RESULTS / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _update_latest_run_link(run_dir: Path) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    latest = RESULTS / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)


def _load_completed_tasks(run_dir: Path) -> dict[str, dict[str, Any]]:
    completed_tasks: dict[str, dict[str, Any]] = {}
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)

    for result_file in task_results_dir.glob("*.json"):
        try:
            completed_tasks[result_file.stem] = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.exception(f"Failed to load completed task results from {result_file}")

    return completed_tasks


def _select_remaining_task_ids(
    task_ids: list[str],
    completed_tasks: Mapping[str, dict[str, Any]],
    limit: int | None,
) -> list[str]:
    remaining_ids = [task_id for task_id in task_ids if task_id not in completed_tasks]
    if limit:
        return remaining_ids[:limit]
    return remaining_ids


def _accumulate_existing_scores(
    completed_tasks: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], int, int, float]:
    all_scores: dict[str, dict[str, Any]] = {}
    total_submitted = 0
    total_tests = 0
    total_cost = 0.0

    for task_id, data in completed_tasks.items():
        score = data.get("score", {})
        all_scores[task_id] = score
        total_submitted += score.get("submitted", 0)
        total_tests += score.get("total", 0)
        total_cost += score.get("total_cost", score.get("cost", 0))

    return all_scores, total_submitted, total_tests, total_cost


def _build_backend_queue(concurrency: int) -> asyncio.Queue[Any] | None:
    if concurrency <= 0:
        return None

    backend_queue: asyncio.Queue[Any] = asyncio.Queue()
    for _ in range(concurrency):
        backend_queue.put_nowait("slot")
    return backend_queue


async def run_all(args: argparse.Namespace) -> None:
    context = OrchestrationContext(
        backend_impl=get_backend_runner(args.backend),
        cli_impl=get_cli_impl(args.cli),
    )
    try:
        context.backend_impl.setup(ROOT, args.cli)
    except DockerException as e:
        logger.error(f"Failed to setup {args.backend}: {e}")
        return

    task_ids = load_task_ids(args.tasks)
    config = TaskProcessConfig.from_args(args)
    run_dir = _resolve_run_dir(args)
    if run_dir is None:
        return

    setup_logging(run_dir)
    logger.info(f"Loaded {len(task_ids)} tasks")

    _update_latest_run_link(run_dir)
    logger.debug(f"Run directory: {run_dir}")

    completed_tasks = _load_completed_tasks(run_dir)
    if completed_tasks:
        logger.info(f"Found {len(completed_tasks)} already-completed tasks, skipping them")

    remaining_ids = _select_remaining_task_ids(task_ids, completed_tasks, getattr(args, "limit", None))
    logger.info(
        f"Running {len(remaining_ids)} tasks ({len(task_ids) - len(completed_tasks) - len(remaining_ids)} skipped)"
    )

    all_scores, total_submitted, total_tests, total_cost = _accumulate_existing_scores(completed_tasks)
    completed = len(completed_tasks)
    backend_queue = _build_backend_queue(getattr(args, "concurrency", 0))

    async def process_and_report(task_id: str) -> None:
        nonlocal completed, total_submitted, total_tests, total_cost
        try:
            result = await process_task(task_id, config, run_dir, backend_queue, context)
        except Exception:
            completed += 1
            logger.exception(f"[{completed}/{len(task_ids)}] CRASH {task_id}")
            return

        score = result["score"]
        total_submitted += score["submitted"]
        total_tests += score["total"]
        total_cost += score.get("total_cost", 0)
        all_scores[task_id] = score
        completed += 1
        submitted = score["submitted"]
        total = score["total"]
        logger.info(
            f"[{completed}/{len(task_ids)}] {'ok' if submitted == total else 'XX'} {task_id}  "
            f"{submitted}/{total} submitted  ({score.get('elapsed', 0):.0f}s)"
        )

    random.shuffle(remaining_ids)
    await asyncio.gather(*(process_and_report(task_id) for task_id in remaining_ids), return_exceptions=True)

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
        "tasks": dict(all_scores),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, encoding="utf-8"))
    logger.info(
        f"\n{'=' * 60}\nDone! {len(task_ids)} tasks, {total_tests} test inputs\nSummary: {run_dir / 'summary.json'}"
    )

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
from typing import Any

from docker.errors import DockerException

from src.backends import get_backend_runner
from src.cli_impl import BaseCLI, get_cli_impl
from src.logger import setup_logging
from src.models import (
    TASK_RESULT_JSON_EXCLUDE,
    AgentConfig,
    AgentResultData,
    CliArgs,
    OrchestrationContext,
    TaskProcessResult,
    TaskScore,
)

ROOT = Path(__file__).resolve().parent
CHALLENGES_FILE = ROOT.parent / "data" / "arc-agi_evaluation_challenges.json"
RESULTS = ROOT.parent / "results"
SESSION_LOG_FILENAME = "session.log"
TRANSCRIPT_FILENAME = "transcript.jsonl"
ATTEMPTS_LOG_FILENAME = "attempts.jsonl"

logger = logging.getLogger(__name__)

_ALL_TASKS: dict[str, dict[str, Any]] = {}


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
    result: AgentResultData,
    task_id: str,
    log_dir: Path,
    cli_impl: BaseCLI,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_lines: list[str] = result.raw_lines

    transcript_path = log_dir / TRANSCRIPT_FILENAME
    if not transcript_path.exists():
        transcript_path.write_text(
            "".join(f"{line}\n" for line in raw_lines),
            encoding="utf-8",
            errors="replace",
        )

    readable_path = log_dir / "readable.md"
    with readable_path.open("w", encoding="utf-8", errors="replace") as readable_file:
        agent_id = result.agent_id
        test_index = result.test_index
        readable_file.write(f"# Agent Log: {agent_id} (task: {task_id}, test: {test_index})\n\n")
        for line in raw_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                readable_file.write(f"[raw] {line}\n")
                continue
            cli_impl.write_readable_log(readable_file, obj)

    attempts_path = log_dir / ATTEMPTS_LOG_FILENAME
    attempts_path.write_text(
        "".join(f"{json.dumps(attempt.model_dump())}\n" for attempt in result.attempts),
        encoding="utf-8",
        errors="replace",
    )

    stderr = result.stderr
    if stderr:
        (log_dir / "stderr.log").write_text(stderr, encoding="utf-8", errors="replace")
    if result.error:
        (log_dir / "error.log").write_text(
            result.error,
            encoding="utf-8",
            errors="replace",
        )


def write_task_result(
    run_dir: Path,
    task_id: str,
    agent_data: AgentResultData,
) -> None:
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    task_file = task_results_dir / f"{task_id}.json"

    if task_file.exists():
        results = TaskProcessResult.model_validate_json(task_file.read_text(encoding="utf-8"))
    else:
        results = TaskProcessResult(task_id=task_id)
    results.update_results(agent_data)
    task_file.write_text(
        json.dumps(results.model_dump(exclude=TASK_RESULT_JSON_EXCLUDE)),
        encoding="utf-8",
    )


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
        for key in os.environ:
            if key.startswith("GEMINI_"):
                envs[key] = os.environ[key]
    elif cli_type == "junie":
        for key in os.environ:
            if key.startswith("JUNIE_"):
                envs[key] = os.environ[key]
    return envs


def _build_agent_configs(
    task_id: str,
    raw_task: dict[str, Any],
    run_dir: Path,
    config: CliArgs,
) -> list[AgentConfig]:
    configs: list[AgentConfig] = []
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

            configs.append(
                AgentConfig(
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
                )
            )

    return configs


async def _run_agent_config(
    config: AgentConfig,
    run_dir: Path,
    backend_semaphore: asyncio.Semaphore,
    context: OrchestrationContext,
) -> AgentResultData:
    try:
        async with backend_semaphore:
            result = await context.backend_impl.start_agent_backend(config)
    except Exception as e:
        # Create error result when backend fails - include empty attempt so it's recorded
        result = AgentResultData(
            task_id=config.task_id,
            agent_id=config.agent_id,
            test_index=config.test_index,
            error=str(e),
        )

    write_agent_logs(result, config.task_id, config.log_dir, context.cli_impl)

    write_task_result(
        run_dir,
        config.task_id,
        result,
    )
    return result


async def process_task(
    task_id: str,
    config: CliArgs,
    run_dir: Path,
    backend_semaphore: asyncio.Semaphore,
    context: OrchestrationContext,
) -> TaskProcessResult:
    raw_task = load_task_json(task_id)
    num_tests = len(raw_task["test"])
    configs = _build_agent_configs(task_id, raw_task, run_dir, config)

    agent_results = await asyncio.gather(
        *[_run_agent_config(config, run_dir, backend_semaphore, context) for config in configs],
    )
    result = TaskProcessResult(task_id=task_id)
    for agent_result in agent_results:
        result.update_results(agent_result)
    result.score.total = num_tests
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    (task_results_dir / f"{task_id}.json").write_text(
        json.dumps(result.model_dump(exclude=TASK_RESULT_JSON_EXCLUDE)),
        encoding="utf-8",
    )
    return result


def _resolve_run_dir(args: CliArgs) -> Path | None:
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
    try:
        latest.symlink_to(run_dir.name)
    except OSError:
        logger.warning(f"Failed to create symlink: {latest} -> {run_dir.name}")


def _load_completed_tasks(run_dir: Path) -> list[TaskProcessResult]:
    completed_tasks: list[TaskProcessResult] = []
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)

    for result_file in task_results_dir.glob("*.json"):
        try:
            completed_tasks.append(TaskProcessResult.model_validate_json(result_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            logger.exception(f"Failed to load completed task results from {result_file}")

    return completed_tasks


def _select_remaining_task_ids(
    task_ids: list[str],
    completed_tasks: list[TaskProcessResult],
    limit: int | None,
) -> list[str]:
    completed_task_ids = [result.task_id for result in completed_tasks]
    remaining_ids = [task_id for task_id in task_ids if task_id not in completed_task_ids]
    if limit is not None:
        return remaining_ids[:limit]
    return remaining_ids


def _accumulate_existing_scores(
    completed_tasks: list[TaskProcessResult],
) -> tuple[dict[str, TaskScore], int, int, float]:
    all_scores: dict[str, TaskScore] = {}
    total_submitted = 0
    total_tests = 0
    total_cost = 0.0

    for result in completed_tasks:
        score = result.score
        all_scores[result.task_id] = score
        total_submitted += score.submitted
        total_tests += score.total
        total_cost += score.total_cost

    return all_scores, total_submitted, total_tests, total_cost


async def run_all(args: CliArgs) -> None:
    task_ids = load_task_ids(args.tasks)
    run_dir = _resolve_run_dir(args)
    if run_dir is None:
        return

    context = OrchestrationContext(
        backend_impl=get_backend_runner(args.backend),
        cli_impl=get_cli_impl(args.cli),
    )
    try:
        context.backend_impl.setup(ROOT, args.cli)
    except DockerException as e:
        logger.error(f"Failed to setup {args.backend}: {e}")
        return
    setup_logging(run_dir)

    logger.info(f"Loaded {len(task_ids)} tasks")

    _update_latest_run_link(run_dir)
    logger.debug(f"Run directory: {run_dir}")

    completed_tasks = _load_completed_tasks(run_dir)
    if completed_tasks:
        logger.info(f"Found {len(completed_tasks)} already-completed tasks, skipping them")

    remaining_ids = _select_remaining_task_ids(task_ids, completed_tasks, args.limit)
    logger.info(
        f"Running {len(remaining_ids)} tasks ({len(task_ids) - len(completed_tasks) - len(remaining_ids)} skipped)"
    )

    all_scores, total_submitted, total_tests, total_cost = _accumulate_existing_scores(completed_tasks)
    completed = len(completed_tasks)
    backend_semaphore = asyncio.Semaphore(args.concurrency)

    async def process_and_report(task_id: str) -> None:
        nonlocal completed, total_submitted, total_tests, total_cost
        try:
            result = await process_task(task_id, args, run_dir, backend_semaphore, context)
        except Exception:
            completed += 1
            logger.exception(f"[{completed}/{len(task_ids)}] CRASH {task_id}")
            return

        score = result.score
        total_submitted += score.submitted
        total_tests += score.total
        total_cost += score.total_cost
        all_scores[task_id] = score
        completed += 1
        logger.info(
            f"[{completed}/{len(task_ids)}] {'ok' if score.submitted == score.total else 'XX'} {task_id}  "
            f"{score.submitted}/{score.total} submitted  ({score.elapsed:.0f}s)"
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
        "whole_task": args.whole_task,
        "num_tasks": len(task_ids),
        "total_tests": total_tests,
        "total_cost": round(total_cost, 2),
        "tasks": {task_id: all_scores[task_id].model_dump() for task_id in all_scores},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        f"\n{'=' * 60}\nDone! {len(task_ids)} tasks, {total_tests} test inputs\nSummary: {run_dir / 'summary.json'}"
    )

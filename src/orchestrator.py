"""Orchestrator: dispatches CLI agents to local Docker containers or E2B sandboxes.

The orchestrator runs locally and handles:
- Task loading from ARC-AGI data files
- Dispatching agents to backends
- Writing logs (raw_stream, transcript, readable, attempts)
- Results aggregation and summary.json
- Resume logic (skip completed tasks)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from src.backends import get_backend_runner
from src.cli_impl import CLIImpl, get_cli_impl
from src.logger import setup_logging

ROOT = Path(__file__).resolve().parent
CHALLENGES_FILE = ROOT.parent / "data" / "arc-agi_evaluation_challenges.json"
RESULTS = ROOT / "results"

_EVENT_FORMATTERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "started": lambda event: f"started (model={event.get('model', '?')})",
    "iteration": lambda event: f"iteration {event.get('iteration', '?')}/{event.get('max_iterations', '?')}",
    "transform_validation": lambda event: (
        f"transform {'PASS' if event.get('all_pass') else 'FAIL'} (iter {event.get('iteration', '?')})"
    ),
    "submitted": lambda event: f"submit #{event.get('attempt', '?')}",
    "done": lambda event: f"done — {event.get('attempts', 0)} attempts, {event.get('elapsed', '?')}s",
    "results_written": lambda _event: "results written",
    "error": lambda event: f"ERROR: {event.get('msg', '')}",
}

logger = logging.getLogger(__name__)

_ALL_TASKS: dict[str, dict[str, Any]] = {}

MAX_RETRIES = 10
INITIAL_BACKOFF_S = 2.0
MAX_BACKOFF_S = 600.0
MAX_EMPTY_RETRIES = 3


class BackendRunner(Protocol):
    async def setup(self, root_path: Path, cli_type: str) -> None: ...
    async def run_agent(
        self,
        task_id: str,
        agent_id: str,
        raw_task: dict[str, Any],
        test_index: int,
        model: str,
        max_iterations: int,
        soft_training_feedback: bool,
        whole_task: bool,
        cli_type: str,
        root_path: Path,
        log_dir: Path,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TaskProcessConfig:
    num_agents: int
    model: str
    max_iterations: int
    soft_training_feedback: bool
    cli: str
    whole_task: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TaskProcessConfig:
        return cls(
            num_agents=args.num_agents,
            model=args.model,
            max_iterations=args.max_iterations,
            soft_training_feedback=args.soft_training_feedback,
            cli=args.cli,
            whole_task=getattr(args, "whole_task", False),
        )


@dataclass(frozen=True)
class AgentRunSpec:
    task_id: str
    agent_id: str
    test_index: int
    log_dir: Path
    raw_task: dict[str, Any]
    model: str
    max_iterations: int
    soft_training_feedback: bool
    whole_task: bool
    cli_type: str
    root_path: Path


@dataclass(frozen=True)
class AgentMeta:
    agent_id: str
    test_index: int
    log_dir: Path


@dataclass(frozen=True)
class AgentAttempt:
    test_index: int
    grid: list[list[int]]


@dataclass(frozen=True)
class UsageTotals:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True)
class AgentResultData:
    agent_id: str
    test_index: int
    attempts: tuple[AgentAttempt, ...]
    cost: float
    backend_cost: float
    backend_duration: float
    total_cost: float
    turns: int
    usage: UsageTotals
    elapsed: float
    error: str | None = None
    raw_lines: tuple[str, ...] = ()
    stderr: str = ""

    @classmethod
    def from_backend_result(cls, result: Mapping[str, Any], fallback_test_index: int) -> AgentResultData:
        attempts: list[AgentAttempt] = []
        for attempt in result.get("attempts", []):
            grid = attempt.get("grid")
            if grid is None:
                continue
            attempts.append(
                AgentAttempt(
                    test_index=attempt.get("test_index", fallback_test_index),
                    grid=grid,
                )
            )

        usage_data = result.get("usage", {})
        return cls(
            agent_id=str(result.get("agent_id", "unknown")),
            test_index=int(result.get("test_index", fallback_test_index)),
            attempts=tuple(attempts),
            cost=float(result.get("cost", 0)),
            backend_cost=float(result.get("backend_cost", 0)),
            backend_duration=float(result.get("backend_duration", 0)),
            total_cost=float(result.get("total_cost", 0)),
            turns=int(result.get("turns", 0)),
            usage=UsageTotals(
                input_tokens=int(usage_data.get("input_tokens", 0)),
                cached_tokens=int(usage_data.get("cached_tokens", 0)),
                output_tokens=int(usage_data.get("output_tokens", 0)),
            ),
            elapsed=float(result.get("elapsed", 0)),
            error=result.get("error"),
            raw_lines=tuple(result.get("raw_lines", [])),
            stderr=str(result.get("stderr", "")),
        )

    @classmethod
    def from_exception(cls, agent_id: str, test_index: int, error: BaseException) -> AgentResultData:
        return cls(
            agent_id=agent_id,
            test_index=test_index,
            attempts=(),
            cost=0.0,
            backend_cost=0.0,
            backend_duration=0.0,
            total_cost=0.0,
            turns=0,
            usage=UsageTotals(),
            elapsed=0.0,
            error=str(error),
        )

    def to_persisted_agent_dict(self) -> dict[str, Any]:
        data = {
            "test_index": self.test_index,
            "attempts": [attempt.grid for attempt in self.attempts],
            "cost": self.cost,
            "backend_cost": self.backend_cost,
            "backend_duration": self.backend_duration,
            "total_cost": self.total_cost,
            "turns": self.turns,
            "usage": self.usage.to_dict(),
        }
        if self.error:
            data["error"] = self.error
        return data

    def submitted_test_indexes(self) -> set[int]:
        return {attempt.test_index for attempt in self.attempts}


@dataclass(frozen=True)
class TaskScore:
    submitted: int
    total: int
    elapsed: float
    api_cost: float
    backend_cost: float
    total_cost: float
    usage: UsageTotals

    def to_dict(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "total": self.total,
            "elapsed": self.elapsed,
            "api_cost": self.api_cost,
            "backend_cost": self.backend_cost,
            "total_cost": self.total_cost,
            "usage": self.usage.to_dict(),
        }


@dataclass(frozen=True)
class TaskProcessResult:
    task_id: str
    score: TaskScore

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "score": self.score.to_dict()}


def _load_all_tasks() -> dict[str, dict[str, Any]]:
    if not _ALL_TASKS:
        if not CHALLENGES_FILE.exists():
            message = f"Challenges file not found: {CHALLENGES_FILE}"
            raise FileNotFoundError(message)
        _ALL_TASKS.update(json.loads(CHALLENGES_FILE.read_text()))
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


def write_agent_logs(result: dict[str, Any], task_id: str, log_dir: Path, cli_impl: CLIImpl) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_lines: list[str] = result.get("raw_lines", [])

    raw_stream_path = log_dir / "raw_stream.jsonl"
    if not raw_stream_path.exists() and raw_lines:
        raw_stream_path.write_text("".join(f"{line}\n" for line in raw_lines))

    transcript_entries = cli_impl.parse_stream_json(raw_lines, task_id)
    transcript_path = log_dir / "transcript.jsonl"
    transcript_path.write_text("".join(f"{json.dumps(entry)}\n" for entry in transcript_entries))

    readable_path = log_dir / "readable.md"
    with readable_path.open("w") as readable_file:
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
    attempts_path.write_text("".join(f"{json.dumps(attempt)}\n" for attempt in result.get("attempts", [])))

    stderr = result.get("stderr", "")
    if stderr:
        (log_dir / "stderr.log").write_text(stderr)
    if "error" in result:
        (log_dir / "error.log").write_text(result["error"])


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
            data = json.loads(task_file.read_text())
        except (json.JSONDecodeError, OSError):
            data = {"agents": {}}
    else:
        data = {"agents": {}}

    data.setdefault("agents", {})[agent_id] = agent_data
    tmp_file.write_text(json.dumps(data, indent=2))
    tmp_file.replace(task_file)


def _build_agent_specs(
    task_id: str,
    raw_task: dict[str, Any],
    run_dir: Path,
    config: TaskProcessConfig,
) -> list[AgentRunSpec]:
    specs: list[AgentRunSpec] = []

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
                    max_iterations=config.max_iterations,
                    soft_training_feedback=config.soft_training_feedback,
                    whole_task=config.whole_task,
                    cli_type=config.cli,
                    root_path=ROOT,
                )
            )

    return specs


async def _run_backend_agent(spec: AgentRunSpec, backend_impl: BackendRunner) -> dict[str, Any]:
    return await backend_impl.run_agent(
        task_id=spec.task_id,
        agent_id=spec.agent_id,
        raw_task=spec.raw_task,
        test_index=spec.test_index,
        model=spec.model,
        max_iterations=spec.max_iterations,
        soft_training_feedback=spec.soft_training_feedback,
        whole_task=spec.whole_task,
        cli_type=spec.cli_type,
        root_path=spec.root_path,
        log_dir=spec.log_dir,
    )


async def _run_agent_with_empty_retry(spec: AgentRunSpec, backend_impl: BackendRunner) -> dict[str, Any]:
    result: dict[str, Any] | None = None

    for empty_attempt in range(MAX_EMPTY_RETRIES + 1):
        result = await _retry_backend_call(
            lambda: _run_backend_agent(spec, backend_impl),
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
    backend_impl: BackendRunner,
    cli_impl: CLIImpl,
) -> AgentResultData:
    if backend_queue is None:
        raw_result = await _run_agent_with_empty_retry(spec, backend_impl)
    else:
        token = await backend_queue.get()
        try:
            raw_result = await _run_agent_with_empty_retry(spec, backend_impl)
        finally:
            backend_queue.put_nowait(token)

    write_agent_logs(raw_result, spec.task_id, spec.log_dir, cli_impl)

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
            (spec.log_dir / "error.log").write_text(str(raw_result))
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
    backend_impl: BackendRunner,
    cli_impl: CLIImpl,
) -> dict[str, Any]:
    config = args if isinstance(args, TaskProcessConfig) else TaskProcessConfig.from_args(args)
    raw_task = load_task_json(task_id)
    num_tests = len(raw_task["test"])
    specs = _build_agent_specs(task_id, raw_task, run_dir, config)

    agent_results = await asyncio.gather(
        *[_run_agent_spec(spec, run_dir, backend_queue, backend_impl, cli_impl) for spec in specs],
        return_exceptions=True,
    )

    task_result, process_result = _collect_task_result(task_id, num_tests, specs, agent_results)
    task_results_dir = run_dir / "task_results"
    task_results_dir.mkdir(exist_ok=True)
    (task_results_dir / f"{task_id}.json").write_text(json.dumps(task_result, indent=2))
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
            completed_tasks[result_file.stem] = json.loads(result_file.read_text())
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


def _write_summary(
    args: argparse.Namespace,
    run_dir: Path,
    task_ids: list[str],
    total_tests: int,
    total_cost: float,
    all_scores: Mapping[str, dict[str, Any]],
) -> None:
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
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        f"\n{'=' * 60}\nDone! {len(task_ids)} tasks, {total_tests} test inputs\nSummary: {run_dir / 'summary.json'}"
    )


async def run_all(args: argparse.Namespace) -> None:
    backend_impl: BackendRunner = get_backend_runner(args.backend)
    cli = get_cli_impl(args.cli)

    try:
        await backend_impl.setup(ROOT, args.cli)
    except Exception as error:
        logger.error(f"Failed to setup {args.backend}: {error}")
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
    logger.info(f"Running {len(remaining_ids)} tasks ({len(task_ids) - len(remaining_ids)} skipped)")

    all_scores, total_submitted, total_tests, total_cost = _accumulate_existing_scores(completed_tasks)
    completed = len(completed_tasks)
    backend_queue = _build_backend_queue(getattr(args, "concurrency", 0))

    async def process_and_report(task_id: str) -> None:
        nonlocal completed, total_submitted, total_tests, total_cost
        try:
            result = await process_task(task_id, config, run_dir, backend_queue, backend_impl, cli)
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

    _write_summary(args, run_dir, task_ids, total_tests, total_cost, all_scores)

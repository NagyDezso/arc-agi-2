from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping
    from pathlib import Path

    from src.backends.base import BackendRunner
    from src.cli_impl import CLIImpl


@dataclass(frozen=True)
class TaskProcessConfig:
    """Configuration for processing a single ARC-AGI task.

    Holds the parameters that control how many agents run, which model/CLI to use,
    and evaluation settings like max iterations and feedback mode.
    """

    num_agents: int
    model: str
    max_iterations: int
    soft_training_feedback: bool
    cli: str
    whole_task: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TaskProcessConfig:
        """Build config from parsed CLI arguments."""
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
    """Specification for a single agent run on one test case.

    One spec is created per (task, agent, test_index) combination. Passed to
    the backend's run_agent to execute the agent.
    """

    task_id: str
    agent_id: str
    test_index: int
    log_dir: Path
    raw_task: dict[str, Any]
    model: str
    envs: dict[str, Any]
    max_iterations: int
    soft_training_feedback: bool
    whole_task: bool
    cli_type: str
    root_path: Path


@dataclass(frozen=True)
class AgentMeta:
    """Minimal metadata identifying an agent run (agent_id, test_index, log_dir)."""

    agent_id: str
    test_index: int
    log_dir: Path


@dataclass(frozen=True)
class AgentAttempt:
    """One submitted solution from an agent.

    The grid is the 2D output matrix the agent produced for the given test_index.
    """

    test_index: int
    grid: list[list[int]]


@dataclass(frozen=True)
class UsageTotals:
    """API token usage for a single agent run (input, cached, output)."""

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        """Serialize for summary.json."""
        return {
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True)
class AgentResultData:
    """Result of one agent run: attempts, costs, usage, and optional error.

    Built from backend output via from_backend_result, or from an exception
    via from_exception when the run fails.
    """

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
        """Parse a backend result dict into AgentResultData."""
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
        """Build a failed result from an exception (no attempts, error message set)."""
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
        """Serialize for writing to summary.json (attempts as grids, usage, costs)."""
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
        """Test indexes for which the agent submitted a solution."""
        return {attempt.test_index for attempt in self.attempts}


@dataclass(frozen=True)
class TaskScore:
    """Aggregated score for a task: submitted/total, elapsed time, costs, usage."""

    submitted: int
    total: int
    elapsed: float
    api_cost: float
    backend_cost: float
    total_cost: float
    usage: UsageTotals

    def to_dict(self) -> dict[str, Any]:
        """Serialize for summary.json."""
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
    """Result of processing one task: task_id plus aggregated TaskScore."""

    task_id: str
    score: TaskScore

    def to_dict(self) -> dict[str, Any]:
        """Serialize for summary.json."""
        return {"task_id": self.task_id, "score": self.score.to_dict()}


@dataclass(frozen=True)
class OrchestrationContext:
    """Runtime context for task processing: backend and CLI implementation to use."""

    backend_impl: BackendRunner
    cli_impl: CLIImpl

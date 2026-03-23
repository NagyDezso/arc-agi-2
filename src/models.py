from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_serializer

if TYPE_CHECKING:
    from backends.base import BackendRunner
    from cli_impl import BaseCLI


class CliArgs(BaseModel):
    tasks: str
    num_agents: int
    max_iterations: int
    model: str
    name: str | None
    resume: str | None
    soft_training_feedback: bool
    whole_task: bool
    concurrency: int
    limit: int | None
    cli: str
    backend: str


class AgentConfig(BaseModel):
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
    envs: dict[str, str]
    max_iterations: int
    soft_training_feedback: bool
    whole_task: bool
    cli_type: str


class AgentAttempt(BaseModel):
    """One submitted solution from an agent.

    The grid is the 2D output matrix the agent produced for the given test_index.
    """

    task_id: str
    attempt: int
    test_index: int
    grid: list[list[int]]


class UsageTotals(BaseModel):
    """API token usage for a single agent run (input, cached, output)."""

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0


class AgentResultData(BaseModel):
    """Result of one agent run: attempts, costs, usage, and optional error."""

    task_id: str
    agent_id: str
    test_index: int
    attempts: list[AgentAttempt] = Field(default_factory=list)
    cost: float = 0.0
    backend_cost: float = 0.0
    backend_duration: float = 0.0
    turns: int = 0
    usage: UsageTotals = Field(default_factory=UsageTotals)
    elapsed: float = 0.0
    error: str | None = Field(default=None, exclude_if=lambda x: x is None)
    raw_lines: list[str] = Field(default_factory=list, exclude=True)
    stderr: str = Field(default="", exclude_if=lambda x: x == "")


class TaskScore(BaseModel):
    """Aggregated score for a task: submitted/total, elapsed time, costs, usage."""

    submitted: int = Field(default=0)
    total: int = Field(default=0)
    elapsed: float = Field(default=0.0)
    api_cost: float = Field(default=0.0)
    backend_cost: float = Field(default=0.0)
    total_cost: float = Field(default=0.0)
    usage: UsageTotals = Field(default_factory=UsageTotals)

    def update_score(self, agent_data: AgentResultData) -> None:
        self.elapsed = max(self.elapsed, agent_data.elapsed)
        self.api_cost += agent_data.cost
        self.backend_cost += agent_data.backend_cost
        self.total_cost += agent_data.cost + agent_data.backend_cost
        self.usage.input_tokens += agent_data.usage.input_tokens
        self.usage.cached_tokens += agent_data.usage.cached_tokens
        self.usage.output_tokens += agent_data.usage.output_tokens


class TaskProcessResult(BaseModel):
    """Result of processing one task: task_id plus aggregated TaskScore."""

    task_id: str
    score: TaskScore = Field(default_factory=TaskScore)
    agents: dict[str, AgentResultData] = Field(default_factory=dict)

    def update_results(self, agent_data: AgentResultData) -> None:
        self.agents[agent_data.agent_id] = agent_data
        submitted_tests: set[int] = set()
        for agent_result in self.agents.values():
            for attempt in agent_result.attempts:
                if attempt.grid:  # Only count non-empty grids as submitted
                    submitted_tests.add(attempt.test_index)
        self.score.submitted = len(submitted_tests)
        self.score.update_score(agent_data)


@dataclass
class OrchestrationContext:
    """Runtime context for task processing: backend and CLI implementation to use."""

    backend_impl: BackendRunner
    cli_impl: BaseCLI

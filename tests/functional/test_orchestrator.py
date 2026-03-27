import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.backends import BackendRunner
from src.models import (
    AgentAttempt,
    AgentConfig,
    AgentResultData,
    CliArgs,
    OrchestrationContext,
    TaskProcessResult,
    TaskScore,
    UsageTotals,
)
from src.orchestrator import SESSION_LOG_FILENAME, TRANSCRIPT_FILENAME, process_task, run_all


class SequenceBackend(BackendRunner):
    def __init__(self, exceptions: list[BaseException] = [], outputs: list[AgentResultData] = []) -> None:
        self._exceptions = list(exceptions)
        self._outputs = list(outputs)
        self.calls: list[str] = []
        self.current_runs = 0
        self.max_concurrent_runs = 0

    def setup(self, root_path: Path, cli_type: str) -> None:
        pass

    async def start_agent_backend(self, spec: AgentConfig) -> AgentResultData:
        self.calls.append(spec.agent_id)
        self.current_runs += 1
        self.max_concurrent_runs = max(self.max_concurrent_runs, self.current_runs)
        try:
            if self._exceptions:
                raise self._exceptions.pop(0)
            output = self._outputs.pop(0)
            log_dir = spec.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / SESSION_LOG_FILENAME).write_text("started\n", encoding="utf-8")
            return output
        finally:
            self.current_runs -= 1


class QueueTrackingBackend(BackendRunner):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []

    def setup(self, root_path: Path, cli_type: str) -> None:
        pass

    async def start_agent_backend(self, spec: AgentConfig) -> AgentResultData:
        self.calls.append(spec.agent_id)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await __import__("asyncio").sleep(0.01)
            test_index = spec.test_index
            return AgentResultData(
                task_id=spec.task_id,
                agent_id=spec.agent_id,
                test_index=test_index,
                attempts=[AgentAttempt(task_id=spec.task_id, attempt=0, test_index=test_index, grid=[[test_index]])],
                elapsed=0.2,
                cost=0.1,
                backend_cost=0.2,
                backend_duration=0.3,
                turns=1,
                usage=UsageTotals(
                    input_tokens=10,
                    cached_tokens=2,
                    output_tokens=3,
                ),
                raw_lines=['{"event": "started"}'],
            )
        finally:
            self.active -= 1


class MockBackend(BackendRunner):
    def setup(self, root_path: Path, cli_type: str) -> None:
        pass

    async def start_agent_backend(self, spec: AgentConfig) -> AgentResultData:
        log_dir = spec.log_dir
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / SESSION_LOG_FILENAME).write_text("started\n", encoding="utf-8")
        return AgentResultData(
            task_id=spec.task_id,
            agent_id=spec.agent_id,
            test_index=spec.test_index,
            attempts=[AgentAttempt(task_id=spec.task_id, attempt=1, test_index=0, grid=[[1, 1], [1, 1]])],
            turns=1,
            cost=0.05,
            backend_cost=0.0,
            backend_duration=1.0,
            usage=UsageTotals(input_tokens=100, cached_tokens=0, output_tokens=50),
            raw_lines=['{"event": "started"}', '{"type": "tool_result", "output": "ok"}'],
            elapsed=1.0,
        )


def _make_args(**overrides: Any) -> CliArgs:
    base = {
        "tasks": "task_a,task_b,task_c",
        "backend": "docker",
        "cli": "opencode",
        "model": "test-model",
        "num_agents": 1,
        "max_iterations": 2,
        "soft_training_feedback": False,
        "whole_task": False,
        "resume": None,
        "name": "testrun",
        "limit": None,
        "concurrency": 2,
    }
    base.update(overrides)
    return CliArgs(**base)


def _task_with_n_tests(count: int) -> dict[str, Any]:
    return {
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[i]]} for i in range(count)],
    }


@pytest.mark.asyncio
@pytest.mark.functional
async def test_process_task_aggregates_multi_agent_multi_test_results(tmp_path: Path, mock_cli_impl) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _make_args(num_agents=2, whole_task=False)

    backend = SequenceBackend(
        exceptions=[],
        outputs=[
            AgentResultData(
                task_id="task_x",
                agent_id="task_x_ens0_t0",
                test_index=0,
                attempts=[AgentAttempt(task_id="task_x", attempt=0, test_index=0, grid=[[1]])],
                elapsed=1.2,
                cost=0.5,
                backend_cost=0.2,
                backend_duration=1.0,
                turns=2,
                usage=UsageTotals(
                    input_tokens=100,
                    cached_tokens=10,
                    output_tokens=20,
                ),
                raw_lines=['{"event": "started"}'],
            ),
            AgentResultData(
                task_id="task_x",
                agent_id="task_x_ens1_t0",
                test_index=0,
                attempts=[AgentAttempt(task_id="task_x", attempt=0, test_index=0, grid=[[2]])],
                elapsed=1.0,
                cost=0.3,
                backend_cost=0.1,
                backend_duration=0.8,
                turns=1,
                usage=UsageTotals(
                    input_tokens=50,
                    cached_tokens=5,
                    output_tokens=7,
                ),
                raw_lines=['{"event": "started"}'],
            ),
            AgentResultData(
                task_id="task_x",
                agent_id="task_x_ens0_t1",
                test_index=1,
                attempts=[AgentAttempt(task_id="task_x", attempt=1, test_index=1, grid=[[3]])],
                elapsed=0.8,
                cost=0.2,
                backend_cost=0.08,
                backend_duration=0.7,
                turns=1,
                usage=UsageTotals(
                    input_tokens=40,
                    cached_tokens=4,
                    output_tokens=6,
                ),
                raw_lines=['{"event": "started"}'],
            ),
            AgentResultData(
                task_id="task_x",
                agent_id="task_x_ens1_t1",
                test_index=1,
                attempts=[AgentAttempt(task_id="task_x", attempt=2, test_index=1, grid=[])],
                elapsed=0.7,
                cost=0.1,
                backend_cost=0.05,
                backend_duration=0.6,
                turns=1,
                usage=UsageTotals(
                    input_tokens=30,
                    cached_tokens=3,
                    output_tokens=4,
                ),
                raw_lines=['{"event": "started"}'],
            ),
        ],
    )

    semaphore = asyncio.Semaphore(1)

    with patch("src.orchestrator.load_task_json", return_value=_task_with_n_tests(2)):
        result = await process_task(
            "task_x",
            args,
            run_dir,
            semaphore,
            OrchestrationContext(backend_impl=backend, cli_impl=mock_cli_impl),
        )

    score = result.score
    assert result.task_id == "task_x"
    assert score.submitted == 2
    assert score.total == 2
    assert score.api_cost == pytest.approx(1.1)
    assert score.backend_cost == pytest.approx(0.43)
    assert score.total_cost == pytest.approx(1.53)
    assert score.elapsed == pytest.approx(1.2)
    assert score.usage == UsageTotals(
        input_tokens=220,
        cached_tokens=22,
        output_tokens=37,
    )
    task_result_path = run_dir / "task_results" / "task_x.json"
    task_result = TaskProcessResult.model_validate_json(task_result_path.read_text(encoding="utf-8"))

    assert set(task_result.agents.keys()) == {
        "task_x_ens0_t0",
        "task_x_ens1_t0",
        "task_x_ens0_t1",
        "task_x_ens1_t1",
    }
    assert task_result.agents["task_x_ens0_t0"].attempts == [
        AgentAttempt(task_id="task_x", attempt=0, test_index=0, grid=[[1]]),
    ]
    assert task_result.agents["task_x_ens0_t1"].attempts == [
        AgentAttempt(task_id="task_x", attempt=1, test_index=1, grid=[[3]]),
    ]
    assert task_result.agents["task_x_ens1_t1"].attempts == [
        AgentAttempt(task_id="task_x", attempt=2, test_index=1, grid=[]),
    ]


@pytest.mark.asyncio
@pytest.mark.functional
async def test_process_task_whole_task_counts_submissions_from_multiple_test_indexes(
    tmp_path: Path,
    mock_cli_impl,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _make_args(num_agents=2, whole_task=True)

    backend = SequenceBackend(
        exceptions=[],
        outputs=[
            AgentResultData(
                task_id="task_whole",
                agent_id="task_whole_ens0",
                test_index=0,
                attempts=[
                    AgentAttempt(task_id="task_whole", attempt=0, test_index=0, grid=[[1]]),
                    AgentAttempt(task_id="task_whole", attempt=1, test_index=1, grid=[[2]]),
                ],
                elapsed=1.5,
                cost=1.0,
                backend_cost=0.5,
                backend_duration=1.4,
                turns=2,
                usage=UsageTotals(
                    input_tokens=40,
                    cached_tokens=4,
                    output_tokens=8,
                ),
                raw_lines=['{"event": "started"}'],
            ),
            AgentResultData(
                task_id="task_whole",
                agent_id="task_whole_ens1",
                test_index=0,
                attempts=[AgentAttempt(task_id="task_whole", attempt=1, test_index=1, grid=[[9]])],
                elapsed=1.0,
                cost=0.4,
                backend_cost=0.2,
                backend_duration=0.9,
                turns=1,
                usage=UsageTotals(input_tokens=10, cached_tokens=1, output_tokens=2),
                raw_lines=['{"event": "started"}'],
            ),
        ],
    )

    semaphore = asyncio.Semaphore(1)

    with patch("src.orchestrator.load_task_json", return_value=_task_with_n_tests(2)):
        result = await process_task(
            "task_whole",
            args,
            run_dir,
            semaphore,
            OrchestrationContext(backend_impl=backend, cli_impl=mock_cli_impl),
        )

    score = result.score
    assert score.submitted == 2
    assert score.total == 2

    readable_log_path = run_dir / "logs" / "task_whole" / "agent0" / "readable.md"
    assert readable_log_path.exists()
    readable_content = readable_log_path.read_text(encoding="utf-8")
    assert "Parsed readable:" in readable_content
    assert "started" in readable_content


@pytest.mark.asyncio
@pytest.mark.functional
async def test_process_task_handles_empty_attempts(tmp_path: Path, mock_cli_impl) -> None:
    """Test that agents with empty attempts don't count toward submitted score."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _make_args(num_agents=1, whole_task=False)

    backend = SequenceBackend(
        exceptions=[],
        outputs=[
            AgentResultData(
                task_id="task_empty",
                agent_id="task_empty_ens0_t0",
                test_index=0,
                attempts=[],  # Empty attempts - no valid submission
                elapsed=0.1,
                cost=0.0,
                backend_cost=0.0,
                backend_duration=0.1,
                turns=0,
                usage=UsageTotals(),
                raw_lines=[],
            ),
        ],
    )

    semaphore = asyncio.Semaphore(1)

    with patch("src.orchestrator.load_task_json", return_value=_task_with_n_tests(1)):
        result = await process_task(
            "task_empty",
            args,
            run_dir,
            semaphore,
            OrchestrationContext(backend_impl=backend, cli_impl=mock_cli_impl),
        )

    score = result.score
    # No submissions since attempts is empty
    assert score.submitted == 0
    assert score.total == 1
    assert len(backend.calls) == 1


@pytest.mark.asyncio
@pytest.mark.functional
async def test_process_task_persists_exception_as_agent_error(tmp_path: Path, mock_cli_impl) -> None:
    """Test that backend exceptions are properly captured and persisted."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _make_args(num_agents=2, whole_task=False)

    # First result is an exception (simulates backend failure for ens0_t0)
    # Second result is successful (for ens1_t0)
    backend = SequenceBackend(
        exceptions=[RuntimeError("backend exploded")],
        outputs=[
            AgentResultData(
                task_id="task_fail",
                agent_id="task_fail_ens1_t0",
                test_index=0,
                attempts=[AgentAttempt(task_id="task_fail", attempt=0, test_index=0, grid=[[5]])],
                elapsed=0.9,
                cost=0.2,
                backend_cost=0.1,
                backend_duration=0.7,
                turns=1,
                usage=UsageTotals(input_tokens=11, cached_tokens=1, output_tokens=2),
                raw_lines=['{"event": "started"}'],
            ),
        ],
    )

    semaphore = asyncio.Semaphore(1)

    with patch("src.orchestrator.load_task_json", return_value=_task_with_n_tests(1)):
        result = await process_task(
            "task_fail",
            args,
            run_dir,
            semaphore,
            OrchestrationContext(backend_impl=backend, cli_impl=mock_cli_impl),
        )

    score = result.score
    assert score.submitted == 1  # Only one successful submission
    task_result_path = run_dir / "task_results" / "task_fail.json"
    task_result = TaskProcessResult.model_validate_json(task_result_path.read_text(encoding="utf-8"))

    # Check that both agents are recorded
    assert "task_fail_ens0_t0" in task_result.agents
    assert "task_fail_ens1_t0" in task_result.agents

    # Failed agent (ens0_t0) should have error and no attempts
    failed_agent = task_result.agents["task_fail_ens0_t0"]
    assert failed_agent.attempts == []
    assert failed_agent.error == "backend exploded"

    # Successful agent (ens1_t0) should have attempts and no error
    ok_agent = task_result.agents["task_fail_ens1_t0"]
    assert ok_agent.attempts == [AgentAttempt(task_id="task_fail", attempt=0, test_index=0, grid=[[5]])]
    assert ok_agent.error is None

    # Error log should exist for failed agent
    error_log_path = run_dir / "logs" / "task_fail" / "t0" / "agent0" / "error.log"
    assert error_log_path.exists()
    assert "backend exploded" in error_log_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.functional
async def test_process_task_respects_backend_semaphore_concurrency_limit(tmp_path: Path, mock_cli_impl) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _make_args(num_agents=3, whole_task=False)
    backend = QueueTrackingBackend()
    semaphore = __import__("asyncio").Semaphore(1)

    with patch("src.orchestrator.load_task_json", return_value=_task_with_n_tests(1)):
        result = await process_task(
            "task_queue",
            args,
            run_dir,
            semaphore,
            OrchestrationContext(backend_impl=backend, cli_impl=mock_cli_impl),
        )

    score = result.score
    assert score.submitted == 1
    assert backend.max_active == 1
    assert len(backend.calls) == 3


@pytest.mark.asyncio
@pytest.mark.functional
async def test_run_all_skips_completed_tasks_and_writes_summary(tmp_path: Path, mock_cli_impl) -> None:
    results_dir = tmp_path / "results"
    resume_dir = results_dir / "resume_run"
    task_results_dir = resume_dir / "task_results"
    task_results_dir.mkdir(parents=True)

    (task_results_dir / "task_a.json").write_text(
        TaskProcessResult(
            task_id="task_a",
            score=TaskScore(
                submitted=1,
                total=1,
                elapsed=0.5,
                api_cost=0.1,
                backend_cost=0.2,
                total_cost=0.3,
                usage=UsageTotals(input_tokens=1, cached_tokens=0, output_tokens=1),
            ),
            agents={},
        ).model_dump_json()
    )

    args = _make_args(
        tasks="task_a,task_b,task_c",
        resume=str(resume_dir),
        limit=1,
        concurrency=2,
        whole_task=False,
    )

    backend = SequenceBackend()
    processed: list[str] = []

    async def fake_process_task(
        task_id: str, args_obj: Any, run_dir: Path, backend_semaphore: asyncio.Semaphore, context: OrchestrationContext
    ) -> TaskProcessResult:
        processed.append(task_id)
        assert run_dir == resume_dir
        assert backend_semaphore is not None
        return TaskProcessResult(
            task_id=task_id,
            score=TaskScore(
                submitted=2,
                total=2,
                elapsed=1.5,
                api_cost=1.0,
                backend_cost=0.5,
                total_cost=1.5,
                usage=UsageTotals(
                    input_tokens=20,
                    cached_tokens=2,
                    output_tokens=4,
                ),
            ),
            agents={
                "task_a_ens0_t0": AgentResultData(
                    task_id="task_a",
                    agent_id="task_a_ens0_t0",
                    test_index=0,
                    attempts=[AgentAttempt(task_id="task_a", attempt=0, test_index=0, grid=[[1]])],
                    elapsed=0.5,
                    cost=0.1,
                    backend_cost=0.1,
                    backend_duration=0.9,
                    turns=1,
                    usage=UsageTotals(
                        input_tokens=10,
                        cached_tokens=1,
                        output_tokens=2,
                    ),
                ),
                "task_a_ens0_t1": AgentResultData(
                    task_id="task_a",
                    agent_id="task_a_ens0_t1",
                    test_index=1,
                    attempts=[AgentAttempt(task_id="task_a", attempt=2, test_index=1, grid=[[2]])],
                    elapsed=1.0,
                    cost=0.2,
                    backend_cost=0.1,
                    backend_duration=0.9,
                    turns=1,
                    usage=UsageTotals(
                        input_tokens=10,
                        cached_tokens=1,
                        output_tokens=2,
                    ),
                ),
            },
        )

    with patch("src.orchestrator.RESULTS", results_dir):
        with patch("src.orchestrator.get_backend_runner", return_value=backend):
            with patch("src.orchestrator.get_cli_impl", return_value=mock_cli_impl):
                with patch(
                    "src.orchestrator.load_task_ids",
                    return_value=["task_a", "task_b", "task_c"],
                ):
                    with patch("src.orchestrator.setup_logging"):
                        with patch("src.orchestrator.process_task", side_effect=fake_process_task):
                            await run_all(args)

    assert processed == ["task_b"]

    summary = json.loads((resume_dir / "summary.json").read_text())
    assert summary["num_tasks"] == 3
    assert summary["total_tests"] == 3
    assert summary["total_cost"] == pytest.approx(1.8)
    assert summary["tasks"]["task_a"]["submitted"] == 1
    assert summary["tasks"]["task_b"]["submitted"] == 2
    assert "task_c" not in summary["tasks"]


@pytest.mark.asyncio
@pytest.mark.functional
async def test_run_all_continues_after_process_task_crash(tmp_path: Path, mock_cli_impl) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    run_dir = results_dir / "testrun_stamp"

    args = _make_args(
        tasks="task_a,task_b",
        name="testrun",
        resume=None,
        limit=None,
        concurrency=0,
    )

    backend = SequenceBackend()

    async def fake_process_task(
        task_id: str,
        args_obj: Any,
        run_dir_arg: Path,
        backend_semaphore: Any,
        context: Any,
    ) -> TaskProcessResult:
        if task_id == "task_a":
            raise RuntimeError("boom")
        return TaskProcessResult(
            task_id=task_id,
            score=TaskScore(
                submitted=1,
                total=1,
                elapsed=0.7,
                api_cost=0.3,
                backend_cost=0.2,
                total_cost=0.5,
                usage=UsageTotals(input_tokens=4, cached_tokens=0, output_tokens=1),
            ),
            agents={},
        )

    with (
        patch("src.orchestrator.RESULTS", results_dir),
        patch("src.orchestrator.get_backend_runner", return_value=backend),
        patch("src.orchestrator.get_cli_impl", return_value=mock_cli_impl),
        patch("src.orchestrator.load_task_ids", return_value=["task_a", "task_b"]),
        patch("src.orchestrator.setup_logging"),
        patch("src.orchestrator.process_task", side_effect=fake_process_task),
        patch("src.orchestrator.random.shuffle", side_effect=lambda items: None),
        patch("src.orchestrator.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value.strftime.return_value = "stamp"
        await run_all(args)

    assert run_dir.exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["num_tasks"] == 2
    assert summary["total_tests"] == 1
    assert summary["total_cost"] == pytest.approx(0.5)
    assert "task_b" in summary["tasks"]
    assert "task_a" not in summary["tasks"]


@pytest.mark.asyncio
@pytest.mark.functional
async def test_process_task_integration(tmp_path, mock_raw_task_file, mock_cli_impl):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    args = CliArgs(
        tasks="test_task_id",
        num_agents=1,
        name="test-name",
        resume=None,
        model="test-model",
        max_iterations=2,
        soft_training_feedback=False,
        whole_task=False,
        concurrency=1,
        limit=None,
        cli="opencode",
        backend="docker",
    )

    backend_impl = MockBackend()

    with patch("src.orchestrator.load_task_json") as mock_load:
        with mock_raw_task_file.open() as f:
            mock_load.return_value = json.load(f)

        result = await process_task(
            "test_task_id",
            args,
            run_dir,
            asyncio.Semaphore(1),
            OrchestrationContext(backend_impl=backend_impl, cli_impl=mock_cli_impl),
        )

    assert result.task_id == "test_task_id"
    assert result.score.submitted == 1
    assert result.score.total == 1

    task_file = run_dir / "task_results" / "test_task_id.json"
    assert task_file.exists()

    log_dir = run_dir / "logs" / "test_task_id" / "t0" / "agent0"
    assert (log_dir / SESSION_LOG_FILENAME).exists()
    assert (log_dir / TRANSCRIPT_FILENAME).exists()
    assert (log_dir / "readable.md").exists()
    assert (log_dir / "attempts.jsonl").exists()

    readable_content = (log_dir / "readable.md").read_text()
    assert "Parsed readable:" in readable_content

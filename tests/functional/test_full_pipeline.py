import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.backends import BackendRunner
from src.models import AgentAttempt, AgentConfig, AgentResultData, CliArgs, UsageTotals
from src.orchestrator import SESSION_LOG_FILENAME, TRANSCRIPT_FILENAME, OrchestrationContext, process_task


class MockBackend(BackendRunner):
    def setup(self, root_path: Path, cli_type: str) -> None:
        pass

    async def start_agent_backend(self, spec: AgentConfig) -> AgentResultData:
        # Simulate a successful run
        log_dir = spec.log_dir
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / SESSION_LOG_FILENAME).write_text("started\n")
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


@pytest.mark.asyncio
@pytest.mark.functional
async def test_process_task_integration(tmp_path, mock_raw_task_file, mock_cli_impl):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    args = CliArgs(
        num_agents=1,
        model="test-model",
        max_iterations=2,
        soft_training_feedback=False,
        cli="opencode",
        whole_task=False,
    )

    backend_impl = MockBackend()

    # Mock load_task_json to return our fixture
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

    # Check result score
    assert result.task_id == "test_task_id"
    assert result.score.submitted == 1
    assert result.score.total == 1

    # Check that task results JSON was written
    task_file = run_dir / "task_results" / "test_task_id.json"
    assert task_file.exists()

    # Check that logs were generated
    log_dir = run_dir / "logs" / "test_task_id" / "t0" / "agent0"
    assert (log_dir / SESSION_LOG_FILENAME).exists()
    assert (log_dir / TRANSCRIPT_FILENAME).exists()
    assert (log_dir / "readable.md").exists()
    assert (log_dir / "attempts.jsonl").exists()

    # Verify readable log content
    readable_content = (log_dir / "readable.md").read_text()
    assert "Parsed readable:" in readable_content

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.cli_impl.base import CLIImpl
from src.orchestrator import OrchestrationContext, process_task


class MockBackend:
    async def run_agent(self, **kwargs):
        # Simulate a successful run
        log_dir = kwargs.get("log_dir")
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "raw_stream.jsonl").write_text('{"event": "started"}')
        return {
            "attempts": [{"test_index": 0, "grid": [[1, 1], [1, 1]]}],
            "turns": 1,
            "cost": 0.05,
            "backend_cost": 0.0,
            "backend_duration": 1.0,
            "total_cost": 0.05,
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "raw_lines": [
                '{"event": "started"}',
                '{"type": "tool_result", "output": "ok"}',
            ],
        }


class MockCLIImpl(CLIImpl):
    def workspace_extras(self, ws_path: Path):
        pass

    def calculate_cost(self, model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        return 0.0

    def run_session(
        self,
        ws_path: Path,
        model: str,
        initial_prompt: str,
        feedback: str,
        iteration: int,
        session_started: bool,
        task_id: str,
        test_index: int,
        _status_cb: Any,
    ) -> tuple[list[str], int, str, dict, bool]:
        return ([], 0, "", {}, False)

    def extract_grid_from_output(self, raw_lines: list[str]) -> list[list[int]] | None:
        return None

    def parse_stream_json(self, raw_lines, task_id):
        return [{"parsed": True, "lines": len(raw_lines)}]

    def write_readable_log(self, rf, line, obj):
        rf.write(f"Parsed readable: {line}\\n")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_task_integration(tmp_path, mock_raw_task_file):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    args = argparse.Namespace(
        num_agents=1,
        model="test-model",
        max_iterations=2,
        soft_training_feedback=False,
        cli="mock",
        whole_task=False,
    )

    backend_impl = MockBackend()
    cli_impl = MockCLIImpl()

    # Mock load_task_json to return our fixture
    with patch("src.orchestrator.load_task_json") as mock_load:
        with open(mock_raw_task_file) as f:
            mock_load.return_value = json.load(f)

        result = await process_task(
            "test_task_id",
            args,
            run_dir,
            None,
            OrchestrationContext(backend_impl=backend_impl, cli_impl=cli_impl),
        )

    # Check result score
    assert result["task_id"] == "test_task_id"
    assert result["score"]["submitted"] == 1
    assert result["score"]["total"] == 1

    # Check that task results JSON was written
    task_file = run_dir / "task_results" / "test_task_id.json"
    assert task_file.exists()

    # Check that logs were generated
    log_dir = run_dir / "logs" / "test_task_id" / "t0" / "agent0"
    assert (log_dir / "raw_stream.jsonl").exists()
    assert (log_dir / "transcript.jsonl").exists()
    assert (log_dir / "readable.md").exists()
    assert (log_dir / "attempts.jsonl").exists()

    # Verify readable log content
    readable_content = (log_dir / "readable.md").read_text()
    assert "Parsed readable:" in readable_content

import logging
from typing import Any
from pathlib import Path
from src.orchestrator import write_agent_logs, _EVENT_FORMATTERS, logger
from src.cli_impl.base import CLIImpl


class MockCLIImpl(CLIImpl):
    def workspace_extras(self, ws_path: Path):
        pass

    def calculate_cost(
        self, model: str, input_tokens: int, cached_tokens: int, output_tokens: int
    ) -> float:
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

    def parse_stream_json(self, raw_lines: list[str], task_id: str) -> list[dict]:
        return [{"parsed": "entry"}]

    def write_readable_log(self, rf: Any, line: str, obj: dict) -> None:
        rf.write("Mock readable log entry\\n")


def test_event_formatters():
    started_formatter = _EVENT_FORMATTERS["started"]
    assert started_formatter({"model": "gpt-4"}) == "started (model=gpt-4)"

    iteration_formatter = _EVENT_FORMATTERS["iteration"]
    assert (
        iteration_formatter({"iteration": 2, "max_iterations": 10}) == "iteration 2/10"
    )


def test_write_agent_logs(tmp_path):
    log_dir = tmp_path / "logs"
    result = {
        "agent_id": "test_agent",
        "test_index": 0,
        "raw_lines": ['{"event": "started"}', "raw log line"],
        "attempts": [{"grid": [[1, 2], [3, 4]]}],
        "stderr": "some error",
    }
    cli_impl = MockCLIImpl()

    write_agent_logs(result, "test_task", log_dir, cli_impl)

    # Check raw_stream.jsonl
    raw_stream_path = log_dir / "raw_stream.jsonl"
    assert raw_stream_path.exists()
    assert '{"event": "started"}' in raw_stream_path.read_text()

    # Check transcript.jsonl
    transcript_path = log_dir / "transcript.jsonl"
    assert transcript_path.exists()
    assert '{"parsed": "entry"}' in transcript_path.read_text()

    # Check readable.md
    readable_path = log_dir / "readable.md"
    assert readable_path.exists()
    assert "Mock readable log entry" in readable_path.read_text()

    # Check attempts.jsonl
    attempts_path = log_dir / "attempts.jsonl"
    assert attempts_path.exists()
    assert '{"grid": [[1, 2], [3, 4]]}' in attempts_path.read_text()

    # Check stderr.log
    stderr_path = log_dir / "stderr.log"
    assert stderr_path.exists()
    assert "some error" in stderr_path.read_text()

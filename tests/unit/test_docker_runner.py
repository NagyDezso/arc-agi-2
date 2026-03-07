import pytest
from src.backends.docker_runner import _handle_status_stdout_line


def test_handle_status_stdout_line(caplog):
    import logging

    caplog.set_level(logging.INFO)

    # Test valid JSON event line
    line = '{"event": "started", "agent_id": "test_agent", "model": "test-model"}'
    _handle_status_stdout_line(line)

    assert any("started (model=test-model)" in record.message for record in caplog.records)
    assert any("[status] test_agent" in record.message for record in caplog.records)


def test_handle_status_stdout_line_invalid_json(caplog):
    line = "Just some standard output not json"
    _handle_status_stdout_line(line)
    assert len(caplog.records) == 0


def test_handle_status_stdout_line_unknown_event(caplog):
    line = '{"event": "unknown_event", "agent_id": "test"}'
    _handle_status_stdout_line(line)
    assert len(caplog.records) == 0

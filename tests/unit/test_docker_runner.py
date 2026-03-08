import logging

from src.backends.docker_runner import DockerRunner


def test_handle_agent_stdout_line(caplog):

    caplog.set_level(logging.INFO)

    line = '{"event": "started", "agent_id": "test_agent", "model": "test-model"}'
    DockerRunner()._handle_agent_stdout_line(line, "arc-agent-123")

    assert any("[agent:arc-agent-123]" in record.message for record in caplog.records)
    assert any("started" in record.message for record in caplog.records)


def test_handle_agent_stdout_line_plain_text(caplog):

    caplog.set_level(logging.INFO)

    line = "Just some standard output not json"
    DockerRunner()._handle_agent_stdout_line(line, "my-container")
    assert any("[agent:my-container]" in record.message for record in caplog.records)

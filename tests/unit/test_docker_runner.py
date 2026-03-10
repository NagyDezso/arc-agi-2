import io
import json
import logging

from src.backends.docker_runner import DockerRunner
from src.log_protocol import encode_status_event, encode_transcript_event


def test_docker_runner_routes_status_and_transcript_lines(caplog):
    runner = DockerRunner()
    session_file = io.StringIO()
    transcript_file = io.StringIO()

    with caplog.at_level(logging.INFO):
        runner._route_agent_output_line(
            encode_status_event("agent started", level="info"),
            session_file,
            transcript_file,
        )
        runner._route_agent_output_line(
            encode_transcript_event({"type": "assistant", "turn": 1, "content": []}),
            session_file,
            transcript_file,
        )
        runner._route_agent_output_line("plain fallback line", session_file, transcript_file)

    assert session_file.getvalue() == "agent started\nplain fallback line\n"
    assert json.loads(transcript_file.getvalue()) == {"type": "assistant", "turn": 1, "content": []}
    assert "agent started" in caplog.text
    assert "plain fallback line" in caplog.text

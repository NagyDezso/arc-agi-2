import io
import logging

from src.backends.docker_runner import DockerRunner
from src.cli_impl import Event, EventType


def test_docker_runner_routes_status_and_transcript_lines(caplog):
    runner = DockerRunner()
    session_file = io.StringIO()
    transcript_file = io.StringIO()

    with caplog.at_level(logging.INFO):
        runner._route_agent_output_line(
            Event(type=EventType.STATUS, message="agent started", level="info").model_dump_json(),
            session_file,
            transcript_file,
        )
        runner._route_agent_output_line(
            Event(type=EventType.TRANSCRIPT, message='{"type":"tool_use","tool_name":"bash"}').model_dump_json(),
            session_file,
            transcript_file,
        )
        runner._route_agent_output_line(
            Event(
                type=EventType.TRANSCRIPT,
                message='{"type":"harness_feedback","for_iteration":1,"text":"fix it"}',
            ).model_dump_json(),
            session_file,
            transcript_file,
        )
        runner._route_agent_output_line("plain fallback line", session_file, transcript_file)

    assert "agent started" in session_file.getvalue()
    assert "Unknown event: plain fallback line" in session_file.getvalue()
    assert transcript_file.getvalue() == (
        '{"type":"tool_use","tool_name":"bash"}\n{"type":"harness_feedback","for_iteration":1,"text":"fix it"}\n'
    )
    assert "agent started" in caplog.text
    assert "Unknown event: plain fallback line" in caplog.text

import logging
from pathlib import Path
from typing import Protocol, TextIO

from pydantic import ValidationError

from src.cli_impl import Event, EventType
from src.models import AgentConfig, AgentResultData

logger = logging.getLogger(__name__)


class BackendRunner(Protocol):
    """Protocol for execution backends (Docker, E2B) that run CLI agents.

    Implementations must provide setup (one-time initialization) and run_agent
    (async execution of a single agent on a task).
    """

    def setup(self, root_path: Path, cli_type: str) -> None:
        """Initialize the backend. Called once before any agent runs."""
        ...

    async def start_agent_backend(
        self,
        config: AgentConfig,
    ) -> AgentResultData:
        """Run one agent on one test case. Returns an AgentResultData with attempts, usage, cost, etc."""
        ...

    def _route_agent_output_line(self, line: str, session_file: TextIO, transcript_file: TextIO) -> None:
        try:
            event = Event.model_validate_json(line)
        except ValidationError as e:
            message = f"Unknown event: {line} - {e}"
            logger.error(message)
            session_file.write(message + "\n")
            session_file.flush()
            return

        if event.type not in {EventType.STATUS, EventType.TRANSCRIPT}:
            logger.info(line)
            session_file.write(line + "\n")
            session_file.flush()
            return

        if event.type == EventType.STATUS:
            message = event.message
            level = logging.getLevelName(event.level.upper())
            if message:
                logger.log(level, message)
                session_file.write(message + "\n")
                session_file.flush()
            return

        transcript_line = event.message
        transcript_file.write(transcript_line + "\n")
        transcript_file.flush()

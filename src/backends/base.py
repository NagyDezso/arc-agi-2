from pathlib import Path
from typing import Any, Protocol

from src.models import AgentRunSpec


class BackendRunner(Protocol):
    """Protocol for execution backends (Docker, E2B) that run CLI agents.

    Implementations must provide setup (one-time initialization) and run_agent
    (async execution of a single agent on a task).
    """

    def setup(self, root_path: Path, cli_type: str) -> None:
        """Initialize the backend. Called once before any agent runs."""
        ...

    async def run_agent(
        self,
        spec: AgentRunSpec,
    ) -> dict[str, Any]:
        """Run one agent on one test case. Returns a result dict with attempts, usage, cost, etc."""
        ...

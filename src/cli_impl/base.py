from typing import Protocol, runtime_checkable, Optional, Any, List
from pathlib import Path


@runtime_checkable
class CLIImpl(Protocol):
    def workspace_extras(self, ws_path: Path) -> None:
        """Applies implementation-specific workspace setup (e.g., settings.json)."""
        ...

    def calculate_cost(self, model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        """Calculates the API cost based on token usage."""
        ...

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
    ) -> tuple[List[str], int, str, dict, bool]:
        """Runs a CLI session and returns (raw_lines, turns, stderr, stats, session_started)."""
        ...

    def extract_grid_from_output(self, raw_lines: List[str]) -> Optional[List[List[int]]]:
        """Extracts the final grid from the session output lines."""
        ...

    def parse_stream_json(self, raw_lines: List[str], task_id: str) -> List[dict]:
        """Parses the raw JSON stream into a transcript format for logging."""
        ...

    def write_readable_log(self, rf: Any, line: str, obj: dict) -> None:
        """Writes a human-readable entry to a log file."""
        ...

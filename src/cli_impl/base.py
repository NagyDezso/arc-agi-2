import json
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

from src.models import UsageTotals

from .types import Event, EventType


def capture_raw_output_line(raw_lines: list[str], raw_line: str) -> dict[str, Any] | None:
    line = raw_line.rstrip("\n").rstrip("\r")
    if not line:
        return None
    raw_lines.append(line)
    print(Event(type=EventType.TRANSCRIPT, message=line).model_dump_json(), flush=True)
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def find_last_grid(text: str) -> list[list[int]] | None:
    if not text:
        return None
    grids = []
    index = 0
    while index < len(text):
        if text[index] == "[" and index + 1 < len(text) and text[index + 1] in "[ \n\r\t":
            depth = 0
            cursor = index
            while cursor < len(text):
                if text[cursor] == "[":
                    depth += 1
                elif text[cursor] == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = text[index : cursor + 1]
                        try:
                            parsed = json.loads(candidate)
                            if (
                                isinstance(parsed, list)
                                and len(parsed) > 0
                                and all(isinstance(row, list) for row in parsed)
                                and all(isinstance(value, int) and 0 <= value <= 9 for row in parsed for value in row)
                            ):
                                grids.append(parsed)
                        except (json.JSONDecodeError, TypeError):
                            pass
                        break
                cursor += 1
        index += 1
    return grids[-1] if grids else None


@runtime_checkable
class BaseCLI(Protocol):
    PRICING: dict[str, tuple[float, float, float]]

    def workspace_extras(self, ws_path: Path) -> None:
        """Applies implementation-specific workspace setup (e.g., settings.json)."""
        ...

    def calculate_cost(self, model: str, usage: UsageTotals) -> float:
        """Calculates the API cost based on token usage."""
        pricing = self.PRICING.get(model)
        if pricing is None:
            return 0.0
        input_rate, output_rate, cached_rate = pricing
        return (
            usage.input_tokens * input_rate / 1_000_000
            + usage.cached_tokens * cached_rate / 1_000_000
            + usage.output_tokens * output_rate / 1_000_000
        )

    def run_session(
        self, ws_path: Path, model: str, initial_prompt: str, feedback: str, iteration: int
    ) -> tuple[list[str], int, str, UsageTotals]:
        """Runs a CLI session and returns (raw_lines, turns, stderr, stats)."""
        ...

    def extract_grid_from_output(self, raw_lines: list[str]) -> list[list[int]] | None:
        """Extracts the final grid from the session output lines."""
        ...

    def write_readable_log(self, rf: TextIO, obj: dict) -> None:
        """Writes a human-readable entry to a log file."""
        ...

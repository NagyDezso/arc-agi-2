"""CLI implementations for opencode, gemini, junie, antigravity, and claude."""

from .antigravity import AntigravityCLI
from .base import BaseCLI
from .claude_code import ClaudeCodeCLI
from .gemini import GeminiCLI
from .junie import JunieCLI
from .opencode import OpenCodeCLI
from .types import Event, EventType

__all__ = ["CLI_IMPLS", "BaseCLI", "Event", "EventType", "get_cli_impl"]

CLI_IMPLS = {
    "opencode": OpenCodeCLI,
    "gemini": GeminiCLI,
    "junie": JunieCLI,
    "antigravity": AntigravityCLI,
    "claude": ClaudeCodeCLI,
}


def get_cli_impl(cli_name: str) -> BaseCLI:
    try:
        return CLI_IMPLS[cli_name]()
    except KeyError:
        raise ValueError(f"Unknown cli name: {cli_name}")

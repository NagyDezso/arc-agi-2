"""CLI implementations for opencode, gemini, and junie."""

from .base import BaseCLI
from .gemini import GeminiCLI
from .junie import JunieCLI
from .opencode import OpenCodeCLI
from .types import Event, EventType

CLI_IMPLS = {
    "opencode": OpenCodeCLI,
    "gemini": GeminiCLI,
    "junie": JunieCLI,
}

def get_cli_impl(cli_name: str) -> BaseCLI:
    try:
        return CLI_IMPLS[cli_name]()
    except KeyError:
        raise ValueError(f"Unknown cli name: {cli_name}")

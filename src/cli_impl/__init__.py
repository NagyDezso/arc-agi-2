"""CLI implementations for open_code and gemini."""

from .base import CLIImpl


def get_cli_impl(cli_name: str) -> CLIImpl:
    if cli_name == "opencode":
        from .opencode import OpenCodeCLI

        return OpenCodeCLI()
    elif cli_name == "gemini":
        from .gemini import GeminiCLI

        return GeminiCLI()
    else:
        raise ValueError(f"Unknown cli name: {cli_name}")

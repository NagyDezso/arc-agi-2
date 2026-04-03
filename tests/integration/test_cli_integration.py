from pathlib import Path

import pytest

from src.cli_impl.gemini import GeminiCLI
from src.cli_impl.junie import JunieCLI
from src.cli_impl.opencode import OpenCodeCLI
from src.models import UsageTotals


def assert_real_usage(raw_lines: list[str], turns: int, stderr: str, stats: UsageTotals) -> None:
    assert not stderr, f"Expected no stderr, got: {stderr}"
    assert turns > 0
    assert stats.input_tokens > 0
    assert stats.output_tokens > 0
    assert len(raw_lines) > 0


@pytest.mark.integration
def test_gemini_real_usage(temp_workspace: Path, mock_test_prompt: str) -> None:
    cli = GeminiCLI()
    raw_lines, turns, stderr, stats = cli.run_session(
        ws_path=temp_workspace, model="gemini-2.5-flash-lite", initial_prompt=mock_test_prompt, feedback="", iteration=0
    )

    assert_real_usage(raw_lines, turns, stderr, stats)


@pytest.mark.integration
def test_opencode_real_usage(temp_workspace: Path, mock_test_prompt: str) -> None:
    cli = OpenCodeCLI()
    raw_lines, turns, stderr, stats = cli.run_session(
        ws_path=temp_workspace, model="opencode/big-pickle", initial_prompt=mock_test_prompt, feedback="", iteration=0
    )
    assert_real_usage(raw_lines, turns, stderr, stats)


@pytest.mark.integration
def test_junie_real_usage(temp_workspace: Path, mock_test_prompt: str) -> None:
    cli = JunieCLI()
    cli.workspace_extras(temp_workspace)
    raw_lines, turns, stderr, stats = cli.run_session(temp_workspace, "gemini-flash", mock_test_prompt, "", 0)
    assert_real_usage(raw_lines, turns, stderr, stats)


def test_opencode_bad_model_failure(temp_workspace: Path):
    cli = OpenCodeCLI()
    raw_lines, turns, stderr, _ = cli.run_session(
        ws_path=temp_workspace,
        model="invalid-model/does-not-exist-12345",
        initial_prompt="Hello, world!",
        feedback="",
        iteration=0,
    )

    assert turns == 0

    error_found = (
        "invalid" in stderr.lower()
        or "not found" in stderr.lower()
        or "error" in stderr.lower()
        or "fail" in stderr.lower()
    )
    if not error_found and len(raw_lines) > 0:
        error_found = any("error" in line.lower() for line in raw_lines)

    assert error_found, f"Expected an error message for invalid model, got stderr: {stderr}"


def test_gemini_bad_model_failure(temp_workspace: Path):
    cli = GeminiCLI()
    cli.workspace_extras(temp_workspace)
    raw_lines, turns, stderr, _ = cli.run_session(
        ws_path=temp_workspace,
        model="gemini-invalid-model-name",
        initial_prompt="Hello, world!",
        feedback="",
        iteration=0,
    )

    assert turns == 0

    error_found = (
        "invalid" in stderr.lower()
        or "not found" in stderr.lower()
        or "error" in stderr.lower()
        or "fail" in stderr.lower()
    )
    if not error_found and len(raw_lines) > 0:
        error_found = any("error" in line.lower() for line in raw_lines)

    assert error_found, f"Expected an error message for invalid model, got stderr: {stderr}"

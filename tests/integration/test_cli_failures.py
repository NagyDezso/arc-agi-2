import pytest
import tempfile
from pathlib import Path
from src.cli_impl.opencode import OpenCodeCLI
from src.cli_impl.gemini import GeminiCLI


def test_opencode_bad_model_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        ws_path = Path(tmpdir)
        cli = OpenCodeCLI()

        # Pass a model that definitely does not exist
        raw_lines, turns, stderr, stats, session_started = cli.run_session(
            ws_path=ws_path,
            model="invalid-model/does-not-exist-12345",
            initial_prompt="Hello, world!",
            feedback="",
            iteration=0,
            session_started=False,
            task_id="test_fail",
            test_index=0,
            _status_cb=lambda e: None,
        )

        # We expect stderr to contain some kind of failure or raw_lines to be empty/error
        # The agent should gracefully return without raising an unhandled exception in python
        assert type(raw_lines) is list
        assert turns == 0

        # The exact error depends on OpenCode, but it should be present in stderr or the process output
        error_found = (
            "invalid" in stderr.lower()
            or "not found" in stderr.lower()
            or "error" in stderr.lower()
            or "fail" in stderr.lower()
        )
        if not error_found and len(raw_lines) > 0:
            error_found = any("error" in line.lower() for line in raw_lines)

        assert error_found, (
            f"Expected an error message for invalid model, got stderr: {stderr}"
        )


def test_gemini_bad_model_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        ws_path = Path(tmpdir)
        cli = GeminiCLI()

        # Needs settings initialized
        cli.workspace_extras(ws_path)

        raw_lines, turns, stderr, stats, session_started = cli.run_session(
            ws_path=ws_path,
            model="gemini-invalid-model-name",
            initial_prompt="Hello, world!",
            feedback="",
            iteration=0,
            session_started=False,
            task_id="test_fail",
            test_index=0,
            _status_cb=lambda e: None,
        )

        assert type(raw_lines) is list
        assert turns == 0

        error_found = (
            "invalid" in stderr.lower()
            or "not found" in stderr.lower()
            or "error" in stderr.lower()
            or "fail" in stderr.lower()
        )
        if not error_found and len(raw_lines) > 0:
            error_found = any("error" in line.lower() for line in raw_lines)

        assert error_found, (
            f"Expected an error message for invalid model, got stderr: {stderr}"
        )

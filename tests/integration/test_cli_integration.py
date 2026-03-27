import tempfile
from pathlib import Path

from src.cli_impl.gemini import GeminiCLI
from src.cli_impl.opencode import OpenCodeCLI


def test_gemini_real_tool_usage():
    with tempfile.TemporaryDirectory() as tmpdir:
        ws_path = Path(tmpdir)
        target_file = ws_path / "gemini_secret.txt"
        target_file.write_text("Gemini successfully read this secret!", encoding="utf-8")

        cli = GeminiCLI()
        model = "gemini-2.5-flash-lite"
        prompt = (
            "Read the file 'gemini_secret.txt' and tell me what is inside. "
            "Then create a file 'gemini_verified.txt' containing the word 'PASSED'."
        )

        raw_lines, _, _, _ = cli.run_session(
            ws_path=ws_path, model=model, initial_prompt=prompt, feedback="", iteration=0
        )
        assert len(raw_lines) > 0

        tool_used = any('"type": "tool_use"' in line or '"type":"tool_use"' in line for line in raw_lines)
        assert tool_used is True, "Gemini did not attempt any tool uses."

        verification_file = ws_path / "gemini_verified.txt"
        if verification_file.exists():
            assert "PASSED" in verification_file.read_text().upper()


def test_opencode_real_tool_usage():
    with tempfile.TemporaryDirectory() as tmpdir:
        ws_path = Path(tmpdir)
        target_file = ws_path / "secret_info.txt"
        target_file.write_text("Hello from the secret integration test file!", encoding="utf-8")

        cli = OpenCodeCLI()
        model = "opencode/big-pickle"
        prompt = (
            "There is a file named 'secret_info.txt' in the current directory. "
            "Please read its contents and write its content exactly into a new file called 'found_secret.txt'. "
            "Do not do anything else."
        )

        raw_lines, _, _, _ = cli.run_session(
            ws_path=ws_path, model=model, initial_prompt=prompt, feedback="", iteration=0
        )

        assert len(raw_lines) > 0

        tool_used = any('"type": "tool_use"' in line or '"type":"tool_use"' in line for line in raw_lines)
        assert tool_used is True, "The model did not attempt any tool uses."

        output_file = ws_path / "found_secret.txt"
        if output_file.exists():
            assert "Hello from the secret integration test file!" in output_file.read_text()


def test_opencode_bad_model_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        ws_path = Path(tmpdir)
        cli = OpenCodeCLI()

        raw_lines, turns, stderr, _ = cli.run_session(
            ws_path=ws_path,
            model="invalid-model/does-not-exist-12345",
            initial_prompt="Hello, world!",
            feedback="",
            iteration=0,
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

        assert error_found, f"Expected an error message for invalid model, got stderr: {stderr}"


def test_gemini_bad_model_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        ws_path = Path(tmpdir)
        cli = GeminiCLI()

        cli.workspace_extras(ws_path)

        raw_lines, turns, stderr, _ = cli.run_session(
            ws_path=ws_path,
            model="gemini-invalid-model-name",
            initial_prompt="Hello, world!",
            feedback="",
            iteration=0,
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

        assert error_found, f"Expected an error message for invalid model, got stderr: {stderr}"

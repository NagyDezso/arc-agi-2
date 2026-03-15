import tempfile
from pathlib import Path

from src.cli_impl.gemini import GeminiCLI


def test_gemini_real_tool_usage():
    # Setup temporary workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        ws_path = Path(tmpdir)
        target_file = ws_path / "gemini_secret.txt"
        target_file.write_text("Gemini successfully read this secret!")

        cli = GeminiCLI()
        # Using a cheap flash model for integration testing
        model = "gemini-2.5-flash-lite"
        prompt = (
            "Read the file 'gemini_secret.txt' and tell me what is inside. "
            "Then create a file 'gemini_verified.txt' containing the word 'PASSED'."
        )

        # Run the session
        raw_lines, _, _, _ = cli.run_session(
            ws_path=ws_path, model=model, initial_prompt=prompt, feedback="", iteration=0
        )
        # Basic assertions that gemini ran
        assert len(raw_lines) > 0

        # Verify tool calls occurred in the raw stream
        tool_used = any('"type": "tool_use"' in line or '"type":"tool_use"' in line for line in raw_lines)

        assert tool_used is True, "Gemini did not attempt any tool uses."

        # Verify side effect
        verification_file = ws_path / "gemini_verified.txt"
        if verification_file.exists():
            assert "PASSED" in verification_file.read_text().upper()

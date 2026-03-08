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
        raw_lines, turns, stderr, stats, session_started = cli.run_session(
            ws_path=ws_path,
            model=model,
            initial_prompt=prompt,
            feedback="",
            iteration=0,
            session_started=False,
            task_id="test_gemini_int",
            test_index=0,
        )
        # Pytest logging of session output
        print("\n[Gemini Session Output]")
        print("raw_lines:")
        for line in raw_lines:
            print(line)
        print(f"turns: {turns}")
        print(f"stderr: {stderr}")
        print(f"stats: {stats}")
        print(f"session_started: {session_started}")

        # Basic assertions that gemini ran
        assert len(raw_lines) > 0

        # Verify tool calls occurred by parsing
        parsed = cli.parse_stream_json(raw_lines, "test_gemini_int")

        tool_used = False
        for entry in parsed:
            if entry.get("type") == "assistant":
                for block in entry.get("content", []):
                    if block.get("type") == "tool_use":
                        tool_used = True
                        break

        assert tool_used is True, "Gemini did not attempt any tool uses."

        # Verify side effect
        verification_file = ws_path / "gemini_verified.txt"
        if verification_file.exists():
            assert "PASSED" in verification_file.read_text().upper()

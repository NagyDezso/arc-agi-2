import pytest
import tempfile
from pathlib import Path
from src.cli_impl.opencode import OpenCodeCLI


def test_opencode_real_tool_usage():
    # Setup temporary workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        ws_path = Path(tmpdir)
        target_file = ws_path / "secret_info.txt"
        target_file.write_text("Hello from the secret integration test file!")

        cli = OpenCodeCLI()
        model = "opencode/big-pickle"
        prompt = (
            "There is a file named 'secret_info.txt' in the current directory. "
            "Please read its contents and write its content exactly into a new file called 'found_secret.txt'. "
            "Do not do anything else."
        )

        # Run the session
        raw_lines, turns, stderr, stats, session_started = cli.run_session(
            ws_path=ws_path,
            model=model,
            initial_prompt=prompt,
            feedback="",
            iteration=0,
            session_started=False,
            task_id="test_int_1",
            test_index=0,
            _status_cb=lambda e: None,
        )

        # Basic assertions that opencode ran
        assert len(raw_lines) > 0

        # Verify tool calls occurred by parsing
        parsed = cli.parse_stream_json(raw_lines, "test_int_1")

        tool_used = False

        for entry in parsed:
            if entry.get("type") == "assistant":
                for block in entry.get("content", []):
                    if block.get("type") == "tool_use":
                        tool_used = True

        assert tool_used is True, "The model did not attempt any tool uses."

        # Check if the desired output file was created
        output_file = ws_path / "found_secret.txt"

        # It's possible the model failed to follow instructions exactly,
        # but as an integration test, we at least expect tool uses and stream parsing to be intact.
        if output_file.exists():
            assert "Hello from the secret integration test file!" in output_file.read_text()

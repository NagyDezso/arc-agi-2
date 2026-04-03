import io
import json

import pytest

from src.cli_impl import get_cli_impl
from src.cli_impl.gemini import GeminiCLI
from src.cli_impl.opencode import OpenCodeCLI


def test_gemini_extract_grid_from_output():
    cli = GeminiCLI()

    # Test valid extraction from tool result text
    raw_lines = [json.dumps({"type": "tool_result", "output": "The answer is [[1, 2], [3, 4]]"})]
    grid = cli.extract_grid_from_output(raw_lines)
    assert grid == [[1, 2], [3, 4]]

    # Test extraction from write_file command
    raw_lines = [
        json.dumps(
            {
                "type": "tool_use",
                "tool_name": "write_file",
                "parameters": {
                    "file_path": "submission.json",
                    "content": "[[5, 5], [5, 5]]",
                },
            }
        )
    ]
    grid = cli.extract_grid_from_output(raw_lines)
    assert grid == [[5, 5], [5, 5]]


def test_get_cli_impl_unknown():
    with pytest.raises(ValueError, match="Unknown cli name"):
        get_cli_impl("not-a-cli")


def test_opencode_extract_grid_from_output():
    cli = OpenCodeCLI()

    # Test text output extraction
    raw_lines = [json.dumps({"type": "text", "part": {"text": "I got the grid: [[9, 8], [7, 6]]"}})]
    grid = cli.extract_grid_from_output(raw_lines)
    assert grid == [[9, 8], [7, 6]]

    # Test tool_use write command extraction
    raw_lines = [
        json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "tool": "write",
                    "state": {
                        "input": {
                            "filePath": "output.json",
                            "content": "[[0, 1], [1, 0]]",
                        }
                    },
                },
            }
        )
    ]
    grid = cli.extract_grid_from_output(raw_lines)
    assert grid == [[0, 1], [1, 0]]


def test_gemini_write_readable_log_handles_split_deltas():
    cli = GeminiCLI()
    output = io.StringIO()
    cli.write_readable_log(
        output,
        {"type": "message", "role": "assistant", "content": "Hello ", "delta": True},
    )
    cli.write_readable_log(
        output,
        {"type": "message", "role": "assistant", "content": "world", "delta": True},
    )
    assert output.getvalue() == "Hello world"


def test_opencode_write_readable_log_includes_tool_result():
    cli = OpenCodeCLI()
    output = io.StringIO()
    cli.write_readable_log(
        output,
        {
            "type": "tool_use",
            "part": {
                "tool": "bash",
                "state": {"input": {"command": "echo test"}, "output": "test output"},
            },
        },
    )
    rendered = output.getvalue()
    assert "**Tool: bash**" in rendered
    assert "**Tool Result:**" in rendered

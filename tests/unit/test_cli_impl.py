import io
import json

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


def test_opencode_calculate_cost():
    cli = OpenCodeCLI()
    # kilo/minimax/minimax-m2.5: $0.29/1M input, $1.20/1M output, $0/1M cache
    cost = cli.calculate_cost(
        "kilo/minimax/minimax-m2.5:free",
        input_tokens=1_000_000,
        cached_tokens=0,
        output_tokens=500_000,
    )
    assert abs(cost - (0.29 + 0.60)) < 0.001  # 0.29 + 0.5*1.20
    assert cli.calculate_cost("unknown-model", 1000, 0, 500) == 0.0


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

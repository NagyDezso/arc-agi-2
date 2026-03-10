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


def test_gemini_parse_stream_json():
    cli = GeminiCLI()
    raw_lines = [
        json.dumps(
            {
                "type": "message",
                "role": "assistant",
                "content": "Let me solve this",
                "delta": False,
            }
        ),
        json.dumps(
            {
                "type": "tool_use",
                "tool_name": "run_shell_command",
                "tool_id": "call_123",
                "parameters": {"command": "ls -l"},
            }
        ),
    ]
    parsed = cli.parse_stream_json(raw_lines, task_id="task_1")
    assert len(parsed) == 1
    assert parsed[0]["type"] == "assistant"
    assert len(parsed[0]["content"]) == 2
    assert parsed[0]["content"][0]["type"] == "text"
    assert parsed[0]["content"][1]["type"] == "tool_use"
    assert parsed[0]["content"][1]["name"] == "Bash"


def test_gemini_parse_stream_json_result_cost():
    cli = GeminiCLI()
    raw_lines = [
        json.dumps({"type": "result", "stats": {"input": 1000, "cached": 100, "output_tokens": 500}}),
    ]
    parsed = cli.parse_stream_json(raw_lines, task_id="t1", model="gemini-2.5-flash")
    assert len(parsed) == 1
    assert parsed[0]["type"] == "result"
    assert parsed[0]["cost"] > 0
    assert parsed[0]["usage"]["input_tokens"] == 1000
    assert parsed[0]["usage"]["cached_tokens"] == 100
    assert parsed[0]["usage"]["output_tokens"] == 500


def test_gemini_build_transcript_stream_handles_split_deltas():
    cli = GeminiCLI()
    stream = cli.build_transcript_stream(task_id="t1", model="gemini-2.5-flash")

    entries = []
    entries.extend(
        stream.consume_raw_line(
            json.dumps({"type": "message", "role": "assistant", "content": "Hello ", "delta": True})
        )
    )
    entries.extend(
        stream.consume_raw_line(json.dumps({"type": "message", "role": "assistant", "content": "world", "delta": True}))
    )
    entries.extend(
        stream.consume_raw_line(
            json.dumps(
                {
                    "type": "tool_use",
                    "tool_name": "run_shell_command",
                    "tool_id": "call_123",
                    "parameters": {"command": "ls -l"},
                }
            )
        )
    )
    entries.extend(stream.finalize())

    assert len(entries) == 1
    assert entries[0]["type"] == "assistant"
    assert entries[0]["content"][0] == {"type": "text", "text": "Hello world"}
    assert entries[0]["content"][1]["type"] == "tool_use"


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


def test_opencode_parse_stream_json():
    cli = OpenCodeCLI()
    raw_lines = [
        json.dumps({"type": "text", "part": {"text": "Running task"}}),
        json.dumps(
            {
                "type": "tool_use",
                "part": {
                    "tool": "bash",
                    "callID": "call_abc",
                    "state": {"input": {"command": "echo test"}, "output": "test\\n"},
                },
            }
        ),
    ]
    parsed = cli.parse_stream_json(raw_lines, task_id="task_1")
    # OpenCode parsing flushes assistant blocks when tool_result equivalent output is found
    # Tool output > 10 chars will trigger user block flush. Since "test\\n" is 5 chars, it just stays in current block until the end
    assert len(parsed) > 0
    assert parsed[0]["type"] == "assistant"


def test_opencode_build_transcript_stream_accumulates_result_until_finalize():
    cli = OpenCodeCLI()
    stream = cli.build_transcript_stream(task_id="task_1", model="kilo/minimax/minimax-m2.5:free")

    entries = []
    entries.extend(stream.consume_raw_line(json.dumps({"type": "text", "part": {"text": "Running task"}})))
    entries.extend(
        stream.consume_raw_line(
            json.dumps(
                {
                    "type": "tool_use",
                    "part": {
                        "tool": "bash",
                        "callID": "call_abc",
                        "state": {"input": {"command": "echo test"}, "output": ""},
                    },
                }
            )
        )
    )
    entries.extend(
        stream.consume_raw_line(
            json.dumps(
                {
                    "type": "step_finish",
                    "part": {"tokens": {"input": 10, "output": 5, "cache": {"read": 2}}},
                }
            )
        )
    )
    entries.extend(stream.finalize())

    assert entries[0]["type"] == "assistant"
    assert entries[1]["type"] == "result"
    assert entries[1]["usage"]["input_tokens"] == 10
    assert entries[1]["usage"]["cached_tokens"] == 2

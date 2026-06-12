import io
import json

import pytest

from src.cli_impl import antigravity as antigravity_mod
from src.cli_impl import get_cli_impl
from src.cli_impl.antigravity import AntigravityCLI
from src.cli_impl.gemini import GeminiCLI
from src.cli_impl.opencode import OpenCodeCLI
from src.models import UsageTotals


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


def test_antigravity_registered():
    assert isinstance(get_cli_impl("antigravity"), AntigravityCLI)


def test_antigravity_extract_grid_from_planner_content():
    cli = AntigravityCLI()
    raw_lines = [json.dumps({"type": "PLANNER_RESPONSE", "content": "Final answer: [[1, 2], [3, 4]]"})]
    assert cli.extract_grid_from_output(raw_lines) == [[1, 2], [3, 4]]


def test_antigravity_extract_grid_from_tool_call_args():
    cli = AntigravityCLI()
    raw_lines = [
        json.dumps(
            {
                "type": "PLANNER_RESPONSE",
                "tool_calls": [
                    {"name": "write_file", "args": {"FilePath": "submission.json", "Content": "[[5, 5], [5, 5]]"}}
                ],
            }
        )
    ]
    assert cli.extract_grid_from_output(raw_lines) == [[5, 5], [5, 5]]


def test_antigravity_write_readable_log_renders_event_types():
    cli = AntigravityCLI()
    output = io.StringIO()
    cli.write_readable_log(output, {"type": "USER_INPUT", "content": "solve the task"})
    cli.write_readable_log(
        output,
        {
            "type": "PLANNER_RESPONSE",
            "content": "Listing files",
            "tool_calls": [{"name": "list_dir", "args": {"DirectoryPath": "/workspace"}}],
        },
    )
    cli.write_readable_log(output, {"type": "CONVERSATION_HISTORY"})
    cli.write_readable_log(output, {"type": "LIST_DIRECTORY", "content": "Empty directory"})
    rendered = output.getvalue()
    assert "**User:**" in rendered
    assert "**Assistant:**" in rendered
    assert "**Tool: list_dir**" in rendered
    assert "**Tool Result (LIST_DIRECTORY):**" in rendered
    # CONVERSATION_HISTORY carries no content and must not be rendered.
    assert "CONVERSATION_HISTORY" not in rendered


def test_antigravity_collect_transcript_events_tracks_offsets(tmp_path, monkeypatch):
    monkeypatch.setattr(antigravity_mod, "_DATA_DIR", tmp_path)
    brain = tmp_path / "brain"
    cli = AntigravityCLI()

    log_dir = brain / "session-1" / ".system_generated" / "logs"
    log_dir.mkdir(parents=True)
    transcript = log_dir / "transcript.jsonl"
    events = [
        {"type": "USER_INPUT", "content": "go"},
        {"type": "PLANNER_RESPONSE", "tool_calls": [{"name": "list_dir", "args": {}}]},
    ]
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    # First call consumes the whole transcript and counts one tool call.
    raw_lines, num_turns = cli._collect_new_transcript_events()
    assert len(raw_lines) == 2
    assert num_turns == 1

    # No new lines -> nothing reported.
    raw_lines, num_turns = cli._collect_new_transcript_events()
    assert raw_lines == []
    assert num_turns == 0

    # Appended events are picked up incrementally (two more tool calls).
    events.append(
        {"type": "PLANNER_RESPONSE", "tool_calls": [{"name": "a", "args": {}}, {"name": "b", "args": {}}]}
    )
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    raw_lines, num_turns = cli._collect_new_transcript_events()
    assert len(raw_lines) == 1
    assert num_turns == 2


def test_antigravity_collect_transcript_events_no_brain_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(antigravity_mod, "_DATA_DIR", tmp_path / "missing")
    cli = AntigravityCLI()
    assert cli._collect_new_transcript_events() == ([], 0)


def _patch_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / ".gemini" / "antigravity-cli"
    monkeypatch.setattr(antigravity_mod, "_DATA_DIR", data_dir)
    return data_dir


def test_antigravity_workspace_extras_writes_settings(tmp_path, monkeypatch):
    data_dir = _patch_data_dir(monkeypatch, tmp_path)
    cli = AntigravityCLI()

    cli.workspace_extras("gemini-3.5-flash")
    settings = json.loads((data_dir / "settings.json").read_text())
    assert settings["model"] == "Gemini 3.5 Flash (Medium)"
    assert settings["trustedWorkspaces"] == ["/workspace"]
    assert settings["enableTelemetry"] is False

    # Unmapped models are written through verbatim.
    cli.workspace_extras("some-other-model")
    settings = json.loads((data_dir / "settings.json").read_text())
    assert settings["model"] == "some-other-model"


def test_antigravity_workspace_extras_registers_statusline(tmp_path, monkeypatch):
    data_dir = _patch_data_dir(monkeypatch, tmp_path)
    # A stale usage log from a previous run must be cleared on setup.
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "usage.jsonl").write_text("stale\n")
    cli = AntigravityCLI()

    cli.workspace_extras("gemini-3.5-flash")

    settings = json.loads((data_dir / "settings.json").read_text())
    script_path = data_dir / "statusline-usage.sh"
    assert settings["statusLine"] == {
        "type": "command",
        "command": str(script_path),
        "enabled": True,
    }
    # The script appends stdin to the usage log and is executable.
    assert str(data_dir / "usage.jsonl") in script_path.read_text()
    assert script_path.stat().st_mode & 0o111
    assert not (data_dir / "usage.jsonl").exists()


def _usage_payload(input_tokens, output_tokens, *, cache_read=0, cache_creation=0):
    return json.dumps(
        {
            "context_window": {
                "current_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_creation_input_tokens": cache_creation,
                    "cache_read_input_tokens": cache_read,
                }
            }
        }
    )


def test_antigravity_collect_usage_groups_streaming_snapshots(tmp_path, monkeypatch):
    data_dir = _patch_data_dir(monkeypatch, tmp_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    usage_log = data_dir / "usage.jsonl"
    # Two requests. The first streams (output climbs 10 -> 40 at constant input),
    # then the second request changes the input and adds a cache read.
    usage_log.write_text(
        "\n".join(
            [
                json.dumps({"context_window": {"current_usage": None}}),  # authenticating
                _usage_payload(1000, 10),
                _usage_payload(1000, 25),
                _usage_payload(1000, 40),
                _usage_payload(2000, 5, cache_read=500, cache_creation=100),
                _usage_payload(2000, 30, cache_read=500, cache_creation=100),
            ]
        )
        + "\n"
    )
    cli = AntigravityCLI()

    usage = cli._collect_usage()

    # input = 1000 + (2000 + 100 creation); output = 40 + 30; cached = 500.
    assert usage.input_tokens == 3100
    assert usage.output_tokens == 70
    assert usage.cached_tokens == 500


def test_antigravity_collect_usage_tracks_offset(tmp_path, monkeypatch):
    data_dir = _patch_data_dir(monkeypatch, tmp_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    usage_log = data_dir / "usage.jsonl"
    cli = AntigravityCLI()

    usage_log.write_text(_usage_payload(1000, 40) + "\n")
    first = cli._collect_usage()
    assert (first.input_tokens, first.output_tokens) == (1000, 40)

    # A second invocation only counts payloads appended since the last read.
    with usage_log.open("a") as fh:
        fh.write(_usage_payload(2000, 50) + "\n")
    second = cli._collect_usage()
    assert (second.input_tokens, second.output_tokens) == (2000, 50)


def test_antigravity_collect_usage_missing_log(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    assert AntigravityCLI()._collect_usage() == UsageTotals()


def test_antigravity_workspace_extras_writes_oauth_token(tmp_path, monkeypatch):
    data_dir = _patch_data_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTIGRAVITY_OAUTH_REFRESH_TOKEN", "refresh-xyz")

    AntigravityCLI().workspace_extras("gemini-3.5-flash")

    token_path = data_dir / "antigravity-oauth-token"
    creds = json.loads(token_path.read_text())
    assert creds["token"]["refresh_token"] == "refresh-xyz"
    assert creds["token"]["token_type"] == "Bearer"
    assert creds["auth_method"] == "consumer"
    # Expiry is in the past so agy refreshes from the refresh token on first use.
    assert creds["token"]["expiry"] == "2000-01-01T00:00:00Z"
    # Credentials file must not be world-readable.
    assert (token_path.stat().st_mode & 0o077) == 0


def test_antigravity_probe_parses_resets_in_from_log_file(tmp_path):
    cli = AntigravityCLI()
    log_path = tmp_path / "agy.log"
    log_path.write_text(
        "I0523 17:12:59.230 282438 printmode.go:130] Print mode: conversation=x, sending message\n"
        "E0523 17:12:59.572 282438 log.go:398] agent executor error: RESOURCE_EXHAUSTED "
        "(code 429): Individual quota reached. Contact your administrator to enable "
        "overages. Resets in 3h45m34s.: RESOURCE_EXHAUSTED (code 429): Individual quota "
        "reached. Contact your administrator to enable overages. Resets in 3h45m34s.\n"
    )
    reset, debug = cli._probe_quota_reset_seconds(log_path)
    assert reset == 3 * 3600 + 45 * 60 + 34
    assert "RESOURCE_EXHAUSTED" in debug


def test_antigravity_probe_parses_seconds_only_reset(tmp_path):
    cli = AntigravityCLI()
    log_path = tmp_path / "agy.log"
    log_path.write_text(
        "E0523 20:58:14.508 519609 log.go:398] agent executor error: RESOURCE_EXHAUSTED "
        "(code 429): Individual quota reached. Resets in 20s.\n"
    )
    reset, _ = cli._probe_quota_reset_seconds(log_path)
    assert reset == 20


def test_antigravity_probe_returns_none_when_log_missing(tmp_path):
    cli = AntigravityCLI()
    reset, debug = cli._probe_quota_reset_seconds(tmp_path / "nope.log")
    assert reset is None
    assert "not found" in debug


def test_antigravity_probe_returns_none_when_no_quota_line(tmp_path):
    cli = AntigravityCLI()
    log_path = tmp_path / "agy.log"
    log_path.write_text("I0523 ... CLI ready\nI0523 ... shutting down\n")
    reset, debug = cli._probe_quota_reset_seconds(log_path)
    assert reset is None
    # Falls back to returning the log tail for diagnostics.
    assert "shutting down" in debug


def test_antigravity_cost_uses_gemini_3_5_flash_pricing():
    cli = AntigravityCLI()
    usage = UsageTotals()
    usage.input_tokens = 1_000_000
    usage.output_tokens = 1_000_000
    usage.cached_tokens = 1_000_000
    # 1.50 input + 9.00 output + 0.15 cached per 1M tokens.
    assert cli.calculate_cost("gemini-3.5-flash", usage) == pytest.approx(10.65)

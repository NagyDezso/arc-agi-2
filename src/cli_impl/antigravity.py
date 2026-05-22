import contextlib
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import TextIO

from src.models import UsageTotals

from .base import BaseCLI, capture_raw_output_line, find_last_grid, strip_ansi

_DATA_DIR = Path("/root/.gemini/antigravity-cli")
# OAuth token file agy reads on startup.
_TOKEN_FILE = _DATA_DIR / "antigravity-oauth-token"
# Each conversation dir holds a structured JSONL transcript we parse.
_BRAIN_DIR = _DATA_DIR / "brain"
_TRANSCRIPT_GLOB = "*/.system_generated/logs/transcript.jsonl"
# agy's settings.json `model` field expects a human-readable display name with
# a reasoning level, not the API id. Only the first entry is verified against a
# real install; others are best-effort and may need correcting.
_MODEL_DISPLAY_NAMES = {
    "gemini-3.5-flash": "Gemini 3.5 Flash (Medium)",
    "claude-sonnet-4-6": "Claude Sonnet 4.6 (Thinking)",
}
# Wall-clock cap on a single `agy --print` invocation.
_SESSION_TIMEOUT_SECONDS = 10800
# Passed to `agy --print-timeout` (Go duration); keep it >= the wall-clock cap.
_PRINT_TIMEOUT = "180m"


class AntigravityCLI(BaseCLI):
    """Google Antigravity CLI (``agy``), the headless successor to the Gemini CLI.

    ``agy --print`` only prints the final plain-text answer and exposes no model
    or structured-output flag. It authenticates from the
    ``~/.gemini/antigravity-cli/antigravity-oauth-token`` file and the model is
    selected via ``settings.json`` (both written by :meth:`workspace_extras`),
    and the real structured record lives in the per-conversation
    ``transcript.jsonl`` files under that same tree; this adapter drives ``agy``
    and then parses the transcript for tool calls, assistant text and the final
    grid. Token usage is not recorded there, so the reported cost is 0.

    See https://antigravity.google/docs/cli-getting-started
    """

    def __init__(self) -> None:
        # Antigravity drives the Gemini model family; rates are USD per 1M
        # tokens as (input, output, cached).
        self.PRICING = {
            "gemini-3.5-flash": (1.50, 9.00, 0.15),
            "gemini-3-flash-preview": (0.50, 3.00, 0.05),
            "gemini-2.5-flash": (0.30, 2.50, 0.03),
            "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
            "gemini-2.5-pro": (1.25, 10.00, 0.125),
            "claude-sonnet-4-6": (3.00, 15.00, 0.30),
        }
        # transcript path -> number of lines already consumed, so each
        # run_session call only reports events appended since the last one.
        self._transcript_offsets: dict[str, int] = {}

    def workspace_extras(self, model: str) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Past expiry makes agy refresh from the refresh token on first use.
        _TOKEN_FILE.write_text(
            json.dumps(
                {
                    "token": {
                        "access_token": "",
                        "token_type": "Bearer",
                        "refresh_token": os.environ.get("ANTIGRAVITY_OAUTH_REFRESH_TOKEN", ""),
                        "expiry": "2000-01-01T00:00:00Z",
                    },
                    "auth_method": "consumer",
                }
            )
        )
        _TOKEN_FILE.chmod(0o600)
        # `agy` has no --model flag; the model is pinned via settings.json.
        (_DATA_DIR / "settings.json").write_text(
            json.dumps(
                {
                    "enableTelemetry": False,
                    "model": _MODEL_DISPLAY_NAMES.get(model, model),
                    # Pre-trust the sandbox workspace so agy never prompts.
                    "trustedWorkspaces": ["/workspace"],
                },
                indent=2,
            )
        )

    def run_session(
        self, ws_path: Path, model: str, initial_prompt: str, feedback: str, iteration: int
    ) -> tuple[list[str], int, str, UsageTotals]:
        base_path = shutil.which("agy")
        if not base_path:
            return [], 0, "agy executable not found in PATH", UsageTotals()

        # Without --add-dir, agy ignores the cwd and works in its own private
        # scratch directory, so transform.py would never land in ws_path.
        cmd = [
            base_path,
            "--dangerously-skip-permissions",
            "--print-timeout",
            _PRINT_TIMEOUT,
            "--add-dir",
            str(ws_path),
        ]
        if iteration == 0:
            cmd.extend(["--print", initial_prompt])
        else:
            cmd.extend(["--continue", "--print", feedback])

        proc = subprocess.Popen(
            cmd,
            cwd=str(ws_path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # isolate process group so we can kill the whole tree
        )
        try:
            _, stderr_text = proc.communicate(timeout=_SESSION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            _, stderr_text = proc.communicate()

        raw_lines, num_turns = self._collect_new_transcript_events()
        # Token usage is absent from the transcript, so cost cannot be computed.
        return raw_lines, num_turns, strip_ansi(stderr_text or ""), UsageTotals()

    def _collect_new_transcript_events(self) -> tuple[list[str], int]:
        """Reads transcript lines appended since the previous call.

        Returns the new JSONL lines (recorded via :func:`capture_raw_output_line`)
        and the number of tool calls they contain.
        """
        raw_lines: list[str] = []
        num_turns = 0
        if not _BRAIN_DIR.is_dir():
            return raw_lines, num_turns

        transcripts = sorted(
            _BRAIN_DIR.glob(_TRANSCRIPT_GLOB),
            key=lambda p: p.stat().st_mtime,
        )
        for path in transcripts:
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            start = self._transcript_offsets.get(str(path), 0)
            for line in lines[start:]:
                obj = capture_raw_output_line(raw_lines, line)
                if obj is not None and isinstance(obj.get("tool_calls"), list):
                    num_turns += len(obj["tool_calls"])
            self._transcript_offsets[str(path)] = len(lines)
        return raw_lines, num_turns

    def extract_grid_from_output(self, raw_lines: list[str]) -> list[list[int]] | None:
        blobs: list[str] = []
        for line in raw_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                blobs.append(line)
                continue
            content = obj.get("content")
            if isinstance(content, str):
                blobs.append(content)
            for tool_call in obj.get("tool_calls") or []:
                if isinstance(tool_call, dict):
                    # A submitted grid may sit in a write/run tool's arguments.
                    blobs.append(json.dumps(tool_call.get("args", {})))
        return find_last_grid("\n".join(blobs))

    def write_readable_log(self, rf: TextIO, obj: dict) -> None:
        evt_type = str(obj.get("type", ""))
        content = obj.get("content")
        if evt_type == "USER_INPUT":
            rf.write(f"\n**User:**\n{content or ''}\n\n")
        elif evt_type == "PLANNER_RESPONSE":
            if isinstance(content, str) and content.strip():
                rf.write(f"\n**Assistant:**\n{content}\n\n")
            for tool_call in obj.get("tool_calls") or []:
                if isinstance(tool_call, dict):
                    name = tool_call.get("name", "")
                    args = json.dumps(tool_call.get("args", {}), indent=2)[:500]
                    rf.write(f"\n**Tool: {name}**\n```\n{args}\n```\n\n")
        elif evt_type == "CONVERSATION_HISTORY":
            return
        elif content:
            rf.write(f"**Tool Result ({evt_type}):**\n```\n{str(content)[:2000]}\n```\n\n")

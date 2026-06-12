import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import TextIO

from src.models import UsageTotals

from .base import BaseCLI, capture_raw_output_line, find_last_grid, strip_ansi
from .types import Event, EventType

_DATA_DIR = Path("/root/.gemini/antigravity-cli")
# Token usage is surfaced only through the statusline hook: agy pipes a JSON
# payload to the statusline command on every render, and that script appends it
# to the usage log. Both live under _DATA_DIR (resolved at call time).
_USAGE_LOG_NAME = "usage.jsonl"
_STATUSLINE_SCRIPT_NAME = "statusline-usage.sh"
# agy's settings.json `model` field expects a human-readable display name with
# a reasoning level, not the API id.
_MODEL_DISPLAY_NAMES = {
    "gemini-3.5-flash": "Gemini 3.5 Flash (Medium)",
    "gemini-3.5-flash-high": "Gemini 3.5 Flash (High)",
    "claude-sonnet-4-6": "Claude Sonnet 4.6 (Thinking)",
}
_NO_INTERNET_PREAMBLE = (
    "STRICT RULE — NO INTERNET ACCESS: You are FORBIDDEN from using any web or "
    "internet tool (search_web, web_search, read_url, read_url_content, or any "
    "browser tool). Looking up the task or its answer online is disqualifying "
    "cheating. Solve the task using ONLY the provided examples and your own "
    "reasoning and local code execution. Never call a web/search/browser tool.\n\n"
)
_SESSION_TIMEOUT_SECONDS = 10800
# Passed to `agy --print-timeout` (Go duration); keep it >= the wall-clock cap.
_PRINT_TIMEOUT = "180m"
# `agy --print` exits silently on quota exhaustion; any run finishing this fast
# with no output is treated as a silent failure.
_QUOTA_SILENT_FAILURE_S = 15.0
# Extra slack on top of the parsed reset window, in case clocks/quotas lag.
_QUOTA_COOLDOWN_HEADROOM_S = 120
# Used when the agy log doesn't expose a parseable "Resets in ..." line.
_QUOTA_FALLBACK_RESET_S = 5 * 3600
# Safety bound so a misdetected silent failure can't spin forever.
_MAX_QUOTA_RETRIES = 3
_QUOTA_RESET_RE = re.compile(
    r"Resets in\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?",
    re.IGNORECASE,
)


class AntigravityCLI(BaseCLI):
    """Google Antigravity CLI (``agy``), the headless successor to the Gemini CLI.

    ``agy --print`` only prints the final plain-text answer and exposes no model
    or structured-output flag. It authenticates from the
    ``~/.gemini/antigravity-cli/antigravity-oauth-token`` file and the model is
    selected via ``settings.json`` (both written by :meth:`workspace_extras`),
    and the real structured record lives in the per-conversation
    ``transcript.jsonl`` files under that same tree; this adapter drives ``agy``
    and then parses the transcript for tool calls, assistant text and the final
    grid. Token usage is not recorded in the transcript, but the statusline hook
    (configured in :meth:`workspace_extras`) does expose per-request counts, so
    cost is reconstructed from the usage log it writes.
    """

    def __init__(self) -> None:
        # USD per 1M tokens: (input, output, cached).
        self.PRICING = {
            "gemini-3.5-flash": (1.50, 9.00, 0.15),
            "gemini-3.5-flash-high": (1.50, 9.00, 0.15),
            "gemini-3-flash-preview": (0.50, 3.00, 0.05),
            "gemini-2.5-flash": (0.30, 2.50, 0.03),
            "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
            "gemini-2.5-pro": (1.25, 10.00, 0.125),
            "claude-sonnet-4-6": (3.00, 15.00, 0.30),
        }
        self._transcript_offsets: dict[str, int] = {}
        self._usage_offset = 0

    def workspace_extras(self, model: str) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        onboarding_file = _DATA_DIR / "cache" / "onboarding.json"
        token_file = _DATA_DIR / "antigravity-oauth-token"
        # Skip agy's first-run wizard (color scheme, ToS, telemetry opt-in)
        # that would otherwise block the headless --print invocation.
        onboarding_file.parent.mkdir(parents=True, exist_ok=True)
        onboarding_file.write_text(
            json.dumps(
                {
                    "consumerOnboardingComplete": True,
                    "enterpriseOnboardingComplete": False,
                    "onboardingComplete": True,
                }
            )
        )
        # Past expiry makes agy refresh from the refresh token on first use.
        token_file.write_text(
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
        token_file.chmod(0o600)
        # agy pipes a statusline JSON payload to this command on stdin on every
        # render; appending it to the usage log is the only way to capture
        # per-request token counts in --print mode.
        usage_log = _DATA_DIR / _USAGE_LOG_NAME
        statusline_script = _DATA_DIR / _STATUSLINE_SCRIPT_NAME
        statusline_script.write_text(f'#!/bin/sh\n{{ cat; printf "\\n"; }} >> "{usage_log}"\n')
        statusline_script.chmod(0o755)
        usage_log.unlink(missing_ok=True)
        self._usage_offset = 0
        # `agy` has no --model flag; the model is pinned via settings.json.
        (_DATA_DIR / "settings.json").write_text(
            json.dumps(
                {
                    "enableTelemetry": False,
                    "model": _MODEL_DISPLAY_NAMES.get(model, model),
                    "statusLine": {
                        "type": "command",
                        "command": str(statusline_script),
                        "enabled": True,
                    },
                    # Pre-trust the sandbox workspace so agy never prompts.
                    "trustedWorkspaces": ["/workspace"],
                },
                indent=2,
            )
        )

    def _emit_status(self, message: str, *, level: str = "info") -> None:
        print(
            Event(type=EventType.STATUS, message=f"[{self.agent_id}] {message}", level=level).model_dump_json(),
            flush=True,
        )

    def run_session(
        self, ws_path: Path, model: str, initial_prompt: str, feedback: str, iteration: int
    ) -> tuple[list[str], int, str, UsageTotals]:
        base_path = shutil.which("agy")
        if not base_path:
            return [], 0, "agy executable not found in PATH", UsageTotals()

        for attempt in range(_MAX_QUOTA_RETRIES + 1):
            raw_lines, num_turns, stderr_text, quota_reset_s = self._invoke_agy(
                base_path, ws_path, initial_prompt, feedback, iteration
            )
            if quota_reset_s is None or attempt == _MAX_QUOTA_RETRIES:
                return raw_lines, num_turns, stderr_text, self._collect_usage()
            cooldown_s = quota_reset_s + _QUOTA_COOLDOWN_HEADROOM_S
            self._emit_status(
                f"agy quota exhausted; sleeping {cooldown_s}s in-process "
                f"(attempt {attempt + 1}/{_MAX_QUOTA_RETRIES}) before retrying"
            )
            time.sleep(cooldown_s)
        return [], 0, "", UsageTotals()  # unreachable

    def _invoke_agy(
        self,
        base_path: str,
        ws_path: Path,
        initial_prompt: str,
        feedback: str,
        iteration: int,
    ) -> tuple[list[str], int, str, int | None]:
        """One agy --print invocation.

        Returns ``(raw_lines, num_turns, stderr_text, quota_reset_s)``. A non-None
        ``quota_reset_s`` indicates a silent quota failure and is the number of
        seconds until the quota resets (falls back to :data:`_QUOTA_FALLBACK_RESET_S`
        when the log doesn't expose a parseable window).
        """
        log_dir = _DATA_DIR / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"orchestrator-{uuid.uuid4().hex}.log"

        # Without --add-dir, agy ignores the cwd and works in its own private
        # scratch directory, so transform.py would never land in ws_path.
        cmd = [
            base_path,
            "--dangerously-skip-permissions",
            "--print-timeout",
            _PRINT_TIMEOUT,
            "--log-file",
            str(log_path),
            "--add-dir",
            str(ws_path),
        ]
        if iteration == 0:
            cmd.extend(["--print", _NO_INTERNET_PREAMBLE + initial_prompt])
        else:
            cmd.extend(["--continue", "--print", _NO_INTERNET_PREAMBLE + feedback])

        proc = subprocess.Popen(
            cmd,
            cwd=str(ws_path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # isolate process group so we can kill the whole tree
        )
        start = time.monotonic()
        try:
            stdout_text, stderr_text = proc.communicate(timeout=_SESSION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            stdout_text, stderr_text = proc.communicate()
        elapsed = time.monotonic() - start

        raw_lines, num_turns = self._collect_new_transcript_events()
        stderr_text = strip_ansi(stderr_text or "")
        stdout_text = strip_ansi(stdout_text or "")

        silent = (
            num_turns == 0
            and not raw_lines
            and not stdout_text.strip()
            and not stderr_text.strip()
            and elapsed < _QUOTA_SILENT_FAILURE_S
        )
        if not silent:
            return raw_lines, num_turns, stderr_text, None

        reset_seconds, probe_debug = self._probe_quota_reset_seconds(log_path)
        if reset_seconds is not None:
            return raw_lines, num_turns, stderr_text, reset_seconds
        self._emit_status(
            f"agy exited silently after {elapsed:.1f}s but the reset window could "
            f"not be parsed; using fallback {_QUOTA_FALLBACK_RESET_S}s. "
            f"[log tail (last 4KB of {log_path.name})]\n{probe_debug or '(empty)'}"
        )
        return raw_lines, num_turns, stderr_text, _QUOTA_FALLBACK_RESET_S

    def _collect_new_transcript_events(self) -> tuple[list[str], int]:
        """Reads transcript lines appended since the previous call.

        Returns the new JSONL lines (recorded via :func:`capture_raw_output_line`)
        and the number of tool calls they contain.
        """
        raw_lines: list[str] = []
        num_turns = 0
        brain_dir = _DATA_DIR / "brain"
        if not brain_dir.is_dir():
            return raw_lines, num_turns

        transcripts = sorted(
            brain_dir.glob("*/.system_generated/logs/transcript.jsonl"),
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

    def _collect_usage(self) -> UsageTotals:
        """Reconstruct billed token usage from statusline payloads since the last call.

        agy re-emits the same ``current_usage`` on every render while a request
        streams, with ``output_tokens`` climbing until the response completes and
        ``input_tokens`` constant. Consecutive payloads sharing the same input are
        therefore one request; its final (max) output is the billed amount. A change
        in the input counts marks the next request.
        """
        try:
            lines = (_DATA_DIR / _USAGE_LOG_NAME).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return UsageTotals()
        start, self._usage_offset = self._usage_offset, len(lines)

        groups: list[list] = []  # [(input, cache_creation, cache_read), max_output]
        for line in lines[start:]:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            usage = (obj.get("context_window") or {}).get("current_usage")
            if not isinstance(usage, dict):
                continue
            key = (
                int(usage.get("input_tokens") or 0),
                int(usage.get("cache_creation_input_tokens") or 0),
                int(usage.get("cache_read_input_tokens") or 0),
            )
            out = int(usage.get("output_tokens") or 0)
            if groups and groups[-1][0] == key:
                groups[-1][1] = max(groups[-1][1], out)
            else:
                groups.append([key, out])

        totals = UsageTotals()
        for (inp, creation, cached), out in groups:
            totals += UsageTotals(
                input_tokens=inp + creation,
                cached_tokens=cached,
                output_tokens=out,
            )
        return totals

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

    def _probe_quota_reset_seconds(self, log_path: Path) -> tuple[int | None, str]:
        """Parse "Resets in Xh Ym Zs" from agy's RESOURCE_EXHAUSTED log line."""
        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return None, f"(log file not found: {log_path})"
        except OSError as exc:
            return None, f"(failed to read log {log_path}: {exc})"

        for line in content.splitlines():
            if "RESOURCE_EXHAUSTED" not in line:
                continue
            match = _QUOTA_RESET_RE.search(line)
            if match:
                hours = int(match.group(1) or 0)
                minutes = int(match.group(2) or 0)
                seconds = int(match.group(3) or 0)
                return hours * 3600 + minutes * 60 + seconds, line.strip()
        return None, content[-4096:]

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

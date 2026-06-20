import contextlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from src.models import UsageTotals

from .base import BaseCLI, capture_raw_output_line, find_last_grid, strip_ansi
from .types import Event, EventType

_CLAUDE_DIR = Path("/root/.claude")
_SESSION_TIMEOUT_SECONDS = 10800
_DENIED_NETWORK_TOOLS = [
    "WebFetch",
    "WebSearch",
    "Bash(curl:*)",
    "Bash(wget:*)",
    "Bash(nc:*)",
    "Bash(ncat:*)",
    "Bash(telnet:*)",
    "Bash(ssh:*)",
    "Bash(scp:*)",
    "Bash(ftp:*)",
    "Bash(pip install:*)",
    "Bash(pip3 install:*)",
    "Bash(npm install:*)",
    "Bash(npm i:*)",
    "Bash(git clone:*)",
    "Bash(git fetch:*)",
    "Bash(git pull:*)",
    "Bash(git push:*)",
]
# Files an agent writes its answer into are scanned for the final grid.
_OUTPUT_FILE_KEYWORDS = ("output", "answer", "result", "solution", "submission")
_WRITE_TOOLS = ("write", "edit", "multiedit")

# --- Session-usage limiter ---------------------------------------------------
_SESSION_LIMIT_ENABLED = os.environ.get("CLAUDE_CODE_SESSION_LIMIT", "1") != "0"
# Wait proactively once utilization reaches this fraction (0..1); 0 waits only on
# a hard block.
_SESSION_LIMIT_THRESHOLD = float(os.environ.get("CLAUDE_CODE_SESSION_LIMIT_THRESHOLD", "0.70"))
_BLOCKING_RL_STATUSES = frozenset({"blocked", "rejected", "exceeded", "exhausted"})
_RESET_HEADROOM_S = 120
_MAX_WAIT_S = 6 * 3600  # cap a single sleep so a bad timestamp can't hang forever


def _seconds_until_epoch(epoch: Any) -> float:
    """Seconds from now until a Unix-epoch-seconds timestamp (0 if past/invalid)."""
    try:
        return max(0.0, float(epoch) - time.time())
    except (TypeError, ValueError):
        return 0.0


class ClaudeCodeCLI(BaseCLI):
    """Anthropic Claude Code CLI (``claude``) in headless ``--print`` mode.

    ``claude --print --output-format stream-json`` emits one JSON event per line:
    a ``system``/``init`` event (session id, model, tools), ``assistant`` and
    ``user`` events carrying message content and per-call ``usage``, and a final
    ``result`` event with the cumulative cost and session id. This adapter drives
    the CLI, streams those events, accumulates token usage, and reuses the session
    across harness iterations via ``--resume <session_id>``.

    Auth is an OAuth token (``claude setup-token``) supplied through the
    ``CLAUDE_CODE_OAUTH_TOKEN`` environment variable, which the CLI reads
    natively — no credential file is written.
    """

    def __init__(self) -> None:
        # USD per 1M tokens: (input, output, cached_read).
        self.PRICING = {
            "opus": (15.00, 75.00, 1.50),
            "claude-opus-4-8": (15.00, 75.00, 1.50),
            "claude-opus-4-6": (15.00, 75.00, 1.50),
            "sonnet": (3.00, 15.00, 0.30),
            "claude-sonnet-4-6": (3.00, 15.00, 0.30),
            "haiku": (1.00, 5.00, 0.10),
            "claude-haiku-4-5": (1.00, 5.00, 0.10),
        }
        self._session_id: str | None = None
        # Latest ``five_hour`` rate_limit_info seen on the CLI stream.
        self._last_five_hour: dict[str, Any] | None = None

    def workspace_extras(self, model: str) -> None:
        _CLAUDE_DIR.mkdir(parents=True, exist_ok=True)

        (_CLAUDE_DIR / "settings.json").write_text(
            json.dumps(
                {
                    "includeCoAuthoredBy": False,
                    "permissions": {
                        "deny": _DENIED_NETWORK_TOOLS,
                    },
                    "env": {
                        "DISABLE_AUTOUPDATER": "1",
                        "DISABLE_TELEMETRY": "1",
                        "DISABLE_ERROR_REPORTING": "1",
                        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _emit_status(self, message: str, *, level: str = "info") -> None:
        print(
            Event(type=EventType.STATUS, message=f"[{self.agent_id}] {message}", level=level).model_dump_json(),
            flush=True,
        )

    def _wait_for_session_capacity(self) -> None:
        """Sleep until ``resetsAt`` when the latest ``five_hour`` event is blocked or
        its utilization is at/above the threshold; otherwise return immediately."""
        if not _SESSION_LIMIT_ENABLED:
            return
        info = self._last_five_hour
        if not info:
            return
        status = info.get("status")
        util = info.get("utilization")
        threshold_pct = _SESSION_LIMIT_THRESHOLD * 100
        over_threshold = threshold_pct > 0 and isinstance(util, (int, float)) and float(util) >= threshold_pct
        if status not in _BLOCKING_RL_STATUSES and not over_threshold:
            return
        reset_s = _seconds_until_epoch(info.get("resetsAt"))
        self._last_five_hour = None  # consume it; the next session re-evaluates
        if reset_s <= 0:
            return
        sleep_s = min(reset_s + _RESET_HEADROOM_S, _MAX_WAIT_S)
        reason = f"status={status}" if status in _BLOCKING_RL_STATUSES else f"utilization {float(util):.0f}%"
        self._emit_status(
            f"5h session limit reached ({reason}); waiting {sleep_s / 60:.0f}m for the window to reset"
        )
        time.sleep(sleep_s)

    def run_session(
        self, ws_path: Path, model: str, initial_prompt: str, feedback: str, iteration: int
    ) -> tuple[list[str], int, str, UsageTotals]:
        base_path = shutil.which("claude")
        if not base_path:
            return [], 0, "claude executable not found in PATH", UsageTotals()

        self._wait_for_session_capacity()

        cmd = [
            base_path,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            model,
            "--dangerously-skip-permissions",
            "--add-dir",
            str(ws_path),
        ]
        if iteration == 0:
            self._session_id = None
            prompt = initial_prompt
        elif self._session_id:
            cmd.extend(["--resume", self._session_id])
            prompt = feedback
        else:
            cmd.append("--continue")
            prompt = feedback
        # The prompt goes on stdin, not as a positional arg: the variadic
        # `--add-dir <directories...>` above would otherwise swallow it.

        raw_lines: list[str] = []
        num_turns = 0
        token_stats = UsageTotals()

        proc = subprocess.Popen(
            cmd,
            cwd=str(ws_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,  # isolate process group so we can kill the whole tree
        )
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise ValueError("Failed to open stdin, stdout or stderr")
        proc.stdin.write(prompt)
        proc.stdin.close()

        line_queue: queue.Queue = queue.Queue()

        def _reader() -> None:
            try:
                for line in proc.stdout or []:
                    if _is_thinking_progress_line(line):
                        continue
                    line_queue.put(line)
            finally:
                line_queue.put(None)

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        def _parse_event(line_str: str) -> None:
            nonlocal num_turns
            obj = capture_raw_output_line(raw_lines, line_str)
            if obj is None:
                return
            session_id = obj.get("session_id")
            if isinstance(session_id, str):
                self._session_id = session_id
            evt_type = obj.get("type")
            if evt_type == "assistant":
                for block in obj.get("message", {}).get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        num_turns += 1
            elif evt_type == "result":
                # The result event carries the authoritative per-turn usage; the
                # per-assistant-message usage is only a streaming snapshot.
                self._accumulate_usage(obj.get("usage"), token_stats)
                turns = obj.get("num_turns")
                if isinstance(turns, int) and turns > num_turns:
                    num_turns = turns
            elif evt_type == "rate_limit_event":
                info = obj.get("rate_limit_info")
                if isinstance(info, dict) and info.get("rateLimitType") == "five_hour":
                    self._last_five_hour = info

        start_time = time.time()
        while True:
            remaining = _SESSION_TIMEOUT_SECONDS - (time.time() - start_time)
            if remaining <= 0:
                proc.terminate()
                break
            try:
                line = line_queue.get(timeout=min(remaining, 60))
            except queue.Empty:
                continue
            if line is None:
                break
            _parse_event(line)

        # Drain whatever the reader buffered after the loop exited.
        while True:
            try:
                line = line_queue.get(timeout=5)
            except queue.Empty:
                break
            if line is None:
                break
            _parse_event(line)

        stderr_text = ""
        try:
            proc.wait(timeout=30)
            stderr_text = proc.stderr.read()
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            with contextlib.suppress(OSError):
                stderr_text = proc.stderr.read()
        stderr_text = strip_ansi(stderr_text or "")
        reader_thread.join(timeout=5)

        return raw_lines, num_turns, stderr_text, token_stats

    @staticmethod
    def _accumulate_usage(usage: Any, token_stats: UsageTotals) -> None:
        """Add a result event's usage. ``iterations`` holds the per-turn counts for
        the requested model; sum it so multi-turn agentic loops are counted in full,
        falling back to the top-level fields for a single turn.

        Cache-creation tokens are billed near input rate, so they fold into input;
        cache-read tokens are the cheap "cached" bucket.
        """
        if not isinstance(usage, dict):
            return
        rows = usage.get("iterations")
        if not isinstance(rows, list) or not rows:
            rows = [usage]
        for row in rows:
            if not isinstance(row, dict):
                continue
            token_stats.input_tokens += int(row.get("input_tokens") or 0) + int(
                row.get("cache_creation_input_tokens") or 0
            )
            token_stats.cached_tokens += int(row.get("cache_read_input_tokens") or 0)
            token_stats.output_tokens += int(row.get("output_tokens") or 0)

    def extract_grid_from_output(self, raw_lines: list[str]) -> list[list[int]] | None:
        write_file_text = ""
        tool_result_text = ""
        assistant_text = ""

        for line in raw_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt_type = obj.get("type", "")
            if evt_type == "assistant":
                for block in obj.get("message", {}).get("content", []) or []:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        assistant_text += str(block.get("text", "")) + "\n"
                    elif btype == "tool_use" and str(block.get("name", "")).lower() in _WRITE_TOOLS:
                        inp = block.get("input", {}) or {}
                        fpath = str(inp.get("file_path", "")).lower()
                        if not fpath.endswith(".py") and any(kw in fpath for kw in _OUTPUT_FILE_KEYWORDS):
                            write_file_text += str(inp.get("content", inp.get("new_string", ""))) + "\n"
            elif evt_type == "user":
                for block in obj.get("message", {}).get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_result_text += _tool_result_text(block.get("content")) + "\n"
            elif evt_type == "result":
                result = obj.get("result")
                if isinstance(result, str):
                    assistant_text += result + "\n"

        for source in (write_file_text, tool_result_text, assistant_text):
            grid = find_last_grid(source)
            if grid is not None:
                return grid
        return None

    def write_readable_log(self, rf: TextIO, obj: dict) -> None:
        evt_type = obj.get("type", "")
        if evt_type == "assistant":
            for block in obj.get("message", {}).get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = str(block.get("text", "")).strip()
                    if text:
                        rf.write(f"\n**Assistant:**\n{text}\n\n")
                elif btype == "tool_use":
                    name = block.get("name", "")
                    inp = block.get("input", {}) or {}
                    if str(name).lower() == "bash":
                        rf.write(f"\n\n**Tool: {name}**\n```\n$ {inp.get('command', '')}\n```\n\n")
                    else:
                        input_str = json.dumps(inp, indent=2)[:500]
                        rf.write(f"\n\n**Tool: {name}**\n```\n{input_str}\n```\n\n")
        elif evt_type == "user":
            for block in obj.get("message", {}).get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    output = _tool_result_text(block.get("content"))[:2000]
                    rf.write(f"**Tool Result:**\n```\n{output}\n```\n\n")
        elif evt_type == "result":
            usage = obj.get("usage", {}) or {}
            total = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
            rf.write(
                f"---\n**Result:** tokens={total}, "
                f"cost=${obj.get('total_cost_usd', '?')}, "
                f"turns={obj.get('num_turns', '?')}\n"
            )
        elif evt_type == "harness_feedback":
            nxt = obj.get("for_iteration", "?")
            body = obj.get("text", "")
            rf.write(f"\n\n**Harness feedback** (next session iteration {nxt}):\n```\n{body}\n```\n\n")


def _is_thinking_progress_line(line: str) -> bool:
    return '"subtype":"thinking_tokens"' in line


def _tool_result_text(content: Any) -> str:
    """Flatten a tool_result ``content`` (str, or list of text/blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(json.dumps(block))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""

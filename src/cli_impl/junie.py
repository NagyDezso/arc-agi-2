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

from .base import BaseCLI, find_last_grid


def _iter_junie_stdout_events(stdout_text: str) -> list[dict[str, Any]]:
    """Parse Junie ``--output-format json`` stdout: one object, a JSON array, or NDJSON lines."""
    text = (stdout_text or "").strip()
    if not text:
        return []
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        events: list[dict[str, Any]] = []
        for raw_line in text.splitlines():
            ln = raw_line.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                events.append(obj)
            elif isinstance(obj, list):
                events.extend(e for e in obj if isinstance(e, dict))
        return events
    if isinstance(parsed, list):
        return [e for e in parsed if isinstance(e, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _parse_transcript_payload(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") != "transcript":
        return None
    message = event.get("message")
    if not isinstance(message, str):
        return None
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _junie_session_payload(event: dict[str, Any]) -> dict[str, Any] | None:
    """Transcript wrapper, or a bare session summary object (``llmUsage`` / ``sessionId``)."""
    wrapped = _parse_transcript_payload(event)
    if wrapped is not None:
        return wrapped
    if isinstance(event.get("llmUsage"), list):
        return event
    if "sessionId" in event and "result" in event:
        return event
    return None


def _summarize_junie_events(
    events: list[dict[str, Any]],
    session_id: str | None,
) -> tuple[list[str], int, UsageTotals, str | None]:
    raw_lines: list[str] = []
    num_turns = 0
    token_stats = UsageTotals()
    next_session_id = session_id

    for event in events:
        raw_lines.append(json.dumps(event, ensure_ascii=False))
        if event.get("type") == "tool_use":
            num_turns += 1
        token_stats.input_tokens += int(event.get("inputTokens") or 0)
        token_stats.output_tokens += int(event.get("outputTokens") or 0)
        token_stats.cached_tokens += int(event.get("cacheInputTokens") or 0)

    for payload in (_junie_session_payload(event) for event in events):
        if payload is None:
            continue
        session_value = payload.get("sessionId")
        if session_value is not None:
            next_session_id = str(session_value)
        result = payload.get("result")
        if isinstance(result, str) and result.strip():
            raw_lines.append(result.strip())
        for row in payload.get("llmUsage") or []:
            if not isinstance(row, dict):
                continue
            token_stats.input_tokens += int(row.get("inputTokens") or 0)
            token_stats.output_tokens += int(row.get("outputTokens") or 0)
            token_stats.cached_tokens += int(row.get("cacheInputTokens") or 0)
            num_turns += int(row.get("calls") or 0)

    return raw_lines, num_turns, token_stats, next_session_id


def _collect_text_blobs(obj: object, out: list[str]) -> None:
    if isinstance(obj, str):
        if obj.strip():
            out.append(obj)
    elif isinstance(obj, dict):
        for key, val in obj.items():
            if key in ("text", "content", "message", "output", "body") and isinstance(val, str):
                out.append(val)
            else:
                _collect_text_blobs(val, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_text_blobs(item, out)


def _drain_junie_stdout_queue(
    line_queue: queue.Queue[str | None], stdout_chunks: list[str], timeout_seconds: float
) -> str | None:
    try:
        line = line_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        return ""
    if line is not None:
        stdout_chunks.append(line)
    return line


def _wait_for_junie_stdout(
    proc: subprocess.Popen[str], line_queue: queue.Queue[str | None], stdout_chunks: list[str], timeout_seconds: int
) -> bool:
    session_start_time = time.time()

    while True:
        remaining = timeout_seconds - (time.time() - session_start_time)
        if remaining <= 0:
            proc.kill()
            return True
        line = _drain_junie_stdout_queue(line_queue, stdout_chunks, min(remaining, 60))
        if line == "":
            if proc.poll() is not None:
                return False
            continue
        if line is None:
            return False


def _drain_remaining_junie_stdout(line_queue: queue.Queue[str | None], stdout_chunks: list[str]) -> None:
    while True:
        line = _drain_junie_stdout_queue(line_queue, stdout_chunks, 1)
        if line in ("", None):
            return


def _collect_junie_stdout(proc: subprocess.Popen[str], timeout_seconds: int) -> tuple[str, bool]:
    if proc.stdout is None:
        return "", False

    line_queue: queue.Queue[str | None] = queue.Queue()
    stdout_chunks: list[str] = []

    def _reader() -> None:
        try:
            for line in proc.stdout:
                line_queue.put(line)
        finally:
            line_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    timed_out = _wait_for_junie_stdout(proc, line_queue, stdout_chunks, timeout_seconds)
    _drain_remaining_junie_stdout(line_queue, stdout_chunks)

    reader_thread.join(timeout=5)
    return "".join(stdout_chunks), timed_out


class JunieCLI(BaseCLI):
    """JetBrains Junie CLI (headless). See https://junie.jetbrains.com/docs/junie-headless.html"""

    def __init__(self) -> None:
        self.PRICING = {
            "gemini-flash": (0.50, 3.00, 0.05),
        }
        self._session_id: str | None = None

    def workspace_extras(self, ws_path: Path) -> None:
        junie_dir = Path("/root/.junie")
        junie_dir.mkdir(parents=True, exist_ok=True)
        jb_account: dict[str, Any] = {
                "jbAccount": {
                    "refresh_token":os.environ.get("JUNIE_REFRESH_TOKEN"),
                    "access_token": os.environ.get("JUNIE_ACCESS_TOKEN") ,
                    "expires_in": 3600,
                    "name": os.environ.get("JUNIE_JB_ACCOUNT_NAME"),
                    "email": os.environ.get("JUNIE_JB_ACCOUNT_EMAIL"),
                }
            }
        (junie_dir / "secure_credentials.json").write_text(json.dumps(jb_account, ensure_ascii=False) + "\n")

    def run_session(
        self, ws_path: Path, model: str, initial_prompt: str, feedback: str, iteration: int
    ) -> tuple[list[str], int, str, UsageTotals]:
        session_timeout = 3600
        base_path = shutil.which("junie")
        if not base_path:
            return [], 0, "junie executable not found in PATH", UsageTotals()

        cmd: list[str] = [
            base_path,
            "--skip-update-check",
            "--output-format",
            "json",
            "--model",
            model,
            "-p",
            str(ws_path),
        ]

        if iteration == 0:
            self._session_id = None
            task_text = initial_prompt
        else:
            if self._session_id:
                cmd.extend(["--session-id", self._session_id])
            task_text = feedback

        cmd.extend(["--task", task_text])
        time.sleep(10000)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ws_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            msg = "Failed to open stdin, stdout or stderr"
            raise ValueError(msg)
        proc.stdin.close()
        stdout_text, timed_out = _collect_junie_stdout(proc, session_timeout)

        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        stderr_text = proc.stderr.read()

        if timed_out:
            token_stats = UsageTotals()
            err = (stderr_text or "") + "\nJunie CLI session timed out."
            return [], 0, err.strip(), token_stats
        events = _iter_junie_stdout_events(stdout_text)
        if not events and (stdout_text or "").strip():
            token_stats = UsageTotals()
            stderr_text += "\nFailed to parse JSON"
            return [], 0, stderr_text, token_stats
        raw_lines, num_turns, token_stats, self._session_id = _summarize_junie_events(events, self._session_id)
        return raw_lines, num_turns, stderr_text, token_stats

    def extract_grid_from_output(self, raw_lines: list[str]) -> list[list[int]] | None:
        blobs: list[str] = []
        for line in raw_lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                blobs.append(line)
                continue
            _collect_text_blobs(obj, blobs)
        combined = "\n".join(blobs)
        return find_last_grid(combined)

    def write_readable_log(self, rf: TextIO, obj: dict[str, Any]) -> None:
        evt_type = str(obj.get("type", ""))
        if evt_type.lower() in ("text", "message", "assistant"):
            text = str(obj.get("text", obj.get("content", "")))
            if text.strip():
                rf.write(f"\n**Assistant:**\n{text.strip()}\n\n")
                return
        snippet = json.dumps(obj, indent=2, ensure_ascii=False)[:2000]
        rf.write(f"\n**Junie ({evt_type or 'event'}):**\n```\n{snippet}\n```\n\n")

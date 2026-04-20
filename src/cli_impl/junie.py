import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from src.models import UsageTotals

from .base import BaseCLI, capture_raw_output_line, find_last_grid


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


def _accumulate_junie_event(
    event: dict[str, Any],
    raw_lines: list[str],
    num_turns: int,
    token_stats: UsageTotals,
    session_id: str | None,
) -> tuple[int, str | None]:
    if event.get("type") == "tool_use":
        num_turns += 1
    token_stats.input_tokens += int(event.get("inputTokens") or 0)
    token_stats.output_tokens += int(event.get("outputTokens") or 0)
    token_stats.cached_tokens += int(event.get("cacheInputTokens") or 0)

    payload = _junie_session_payload(event)
    if payload is None:
        return num_turns, session_id
    session_value = payload.get("sessionId")
    if session_value is not None:
        session_id = str(session_value)
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

    return num_turns, session_id


def _stream_junie_stdout(
    lines: Iterable[str],
    raw_lines: list[str],
    session_id: str | None,
) -> tuple[int, UsageTotals, str | None, bool]:
    num_turns = 0
    token_stats = UsageTotals()
    parsed_any = False
    for line in lines:
        obj = capture_raw_output_line(raw_lines, line)
        if obj is not None:
            num_turns, session_id = _accumulate_junie_event(obj, raw_lines, num_turns, token_stats, session_id)
            parsed_any = True
            continue
        stripped = line.rstrip("\n").rstrip("\r")
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            for e in parsed:
                if isinstance(e, dict):
                    num_turns, session_id = _accumulate_junie_event(e, raw_lines, num_turns, token_stats, session_id)
                    parsed_any = True
    return num_turns, token_stats, session_id, parsed_any


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


class JunieCLI(BaseCLI):
    """JetBrains Junie CLI (headless). See https://junie.jetbrains.com/docs/junie-headless.html"""

    def __init__(self) -> None:
        self.PRICING = {
            "gemini-flash": (0.50, 3.00, 0.05),
        }
        self._session_id: str | None = None

    def workspace_extras(self) -> None:
        junie_dir = Path("/root/.junie")
        junie_dir.mkdir(parents=True, exist_ok=True)
        (junie_dir / "secure_credentials.json").write_text(
            json.dumps(
                {
                    "secrets": [
                        {
                            "key": "jb-account-stored",
                            "secret": json.dumps(
                                {
                                    "jbAccount": {
                                        "refresh_token": os.environ.get("JUNIE_REFRESH_TOKEN"),
                                        "access_token": os.environ.get("JUNIE_ACCESS_TOKEN"),
                                        "expires_in": 3600,
                                        "name": os.environ.get("JUNIE_JB_ACCOUNT_NAME"),
                                        "email": os.environ.get("JUNIE_JB_ACCOUNT_EMAIL"),
                                    }
                                }
                            ),
                        }
                    ]
                },
                indent=2,
            )
        )

    def run_session(
        self, ws_path: Path, model: str, initial_prompt: str, feedback: str, iteration: int
    ) -> tuple[list[str], int, str, UsageTotals]:
        session_timeout = 3600
        base_path = shutil.which("junie")
        if not base_path:
            return [], 0, "junie executable not found in PATH", UsageTotals()

        cmd: list[str] = [base_path,"--skip-update-check","--output-format","json","--model",model]
        if iteration == 0:
            self._session_id = None
            task_text = initial_prompt
        else:
            if self._session_id:
                cmd.extend(["--session-id", self._session_id])
            task_text = feedback
        time.sleep(1000)
        cmd.extend(["--task", task_text])
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

        raw_lines: list[str] = []
        num_turns, token_stats, self._session_id, parsed_any = _stream_junie_stdout(
            proc.stdout or [], raw_lines, self._session_id
        )

        stderr_text = proc.stderr.read()
        timed_out = False
        try:
            proc.wait(timeout=session_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True
            stderr_text = (stderr_text or "") + proc.stderr.read()

        if timed_out:
            token_stats = UsageTotals()
            err = (stderr_text or "") + "\nJunie CLI session timed out."
            return [], 0, err.strip(), token_stats
        if not parsed_any and raw_lines:
            token_stats = UsageTotals()
            stderr_text = (stderr_text or "") + "\nFailed to parse JSON"
            return [], 0, stderr_text, token_stats

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

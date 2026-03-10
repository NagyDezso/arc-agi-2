"""Shared log-stream protocol helpers for agent stdout routing."""

from __future__ import annotations

import json
from typing import Any

SESSION_LOG_FILENAME = "session.log"
TRANSCRIPT_FILENAME = "transcript.jsonl"

STATUS_EVENT = "status"
TRANSCRIPT_EVENT = "transcript"


def encode_status_event(message: str, *, level: str = "info") -> str:
    return json.dumps({"type": STATUS_EVENT, "level": level, "message": message})


def encode_transcript_event(entry: dict[str, Any]) -> str:
    return json.dumps({"type": TRANSCRIPT_EVENT, "entry": entry})


def decode_stream_event(line: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    event_type = obj.get("type")
    if event_type not in {STATUS_EVENT, TRANSCRIPT_EVENT}:
        return None
    return obj

"""Event types for CLI implementations."""

import enum

from pydantic import BaseModel


class EventType(enum.StrEnum):
    STATUS = "status"
    TRANSCRIPT = "transcript"


class Event(BaseModel):
    type: EventType
    message: str
    level: str = "info"

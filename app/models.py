from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

EVENT_TYPE_CATALOGUE = (
    "ENTRY",
    "EXIT",
    "ZONE_ENTER",
    "ZONE_EXIT",
    "ZONE_DWELL",
    "BILLING_QUEUE_JOIN",
    "BILLING_QUEUE_ABANDON",
    "REENTRY",
)


class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None


class EventEnvelope(BaseModel):
    event_id: str
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: str
    timestamp: str
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool
    confidence: float
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("event_id")
    @classmethod

    def validate_uuid4(cls, value: str) -> str:
        try:
            parsed = UUID(str(value))
        except Exception as exc:
            raise ValueError("event_id_must_be_uuidv4") from exc
        if parsed.version != 4:
            raise ValueError("event_id_must_be_uuidv4")
        return str(parsed)

    @field_validator("event_type")
    @classmethod

    def validate_event_type(cls, value: str) -> str:
        canonical = str(value).strip().upper()
        if canonical not in EVENT_TYPE_CATALOGUE:
            raise ValueError("unsupported_event_type")
        return canonical

    @field_validator("timestamp")
    @classmethod

    def validate_timestamp(cls, value: str) -> str:
        text = str(value).strip()
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timestamp_must_be_utc")
        utc_value = parsed.astimezone(timezone.utc)
        return utc_value.isoformat().replace("+00:00", "Z")

    @field_validator("zone_id")
    @classmethod

    def validate_zone_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() == "none":
            return None
        return text

    @field_validator("dwell_ms")
    @classmethod

    def validate_dwell_ms(cls, value: int) -> int:
        return int(value)

    @field_validator("metadata")
    @classmethod

    def validate_metadata(cls, value: Dict[str, Any] | EventMetadata) -> EventMetadata:
        if isinstance(value, EventMetadata):
            return value
        if not isinstance(value, dict):
            raise ValueError("metadata_must_be_object")
        return EventMetadata.model_validate(value)

InboundEventModel = EventEnvelope

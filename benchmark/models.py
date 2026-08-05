from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


SCHEMA_VERSION = "2.0"


class EventType(StrEnum):
    REQUEST_STARTED = "request_started"
    CONNECTION_STARTED = "connection_started"
    CONNECTION_READY = "connection_ready"
    REQUEST_BODY_SENT = "request_body_sent"
    RESPONSE_HEADERS = "response_headers_received"
    FIRST_RESPONSE_BYTE = "first_response_byte"
    AUDIO_CHUNK = "audio_chunk"
    SERVER_TIMING = "server_timing"
    STREAM_FINISHED = "stream_finished"
    REQUEST_FAILED = "request_failed"


@dataclass(frozen=True)
class AudioFormat:
    encoding: str
    sample_rate_hz: int
    channels: int = 1
    sample_width_bytes: int = 2


@dataclass
class ProviderEvent:
    type: EventType
    at_ns: int
    data: dict[str, Any] = field(default_factory=dict)
    audio: bytes | None = None
    audio_format: AudioFormat | None = None

    def public_dict(self, started_ns: int) -> dict[str, Any]:
        item = {
            "type": self.type.value,
            "offset_ms": (self.at_ns - started_ns) / 1_000_000,
            "data": self.data,
        }
        if self.audio is not None:
            item["audio_bytes"] = len(self.audio)
        if self.audio_format is not None:
            item["audio_format"] = asdict(self.audio_format)
        return item


@dataclass(frozen=True)
class SynthesisRequest:
    request_id: str
    text: str
    language: str | None = None
    voice: str | None = None


@dataclass
class SampleResult:
    run_id: str
    sample_id: str
    client_id: str
    client_region: str
    provider_id: str
    provider_kind: str
    profile_id: str
    text_id: str
    text: str
    connection_mode: str
    transport: str
    protocol: str
    status: str
    started_at: str
    ttfb_ms: float | None = None
    initial_silence_ms: float | None = None
    initial_silence_v1_ms: float | None = None
    initial_silence_v2_ms: float | None = None
    ttfa_ms: float | None = None
    ttfa_v1_ms: float | None = None
    ttfa_v2_ms: float | None = None
    ttfa_audio_clock_ms: float | None = None
    ttfa_available_ms: float | None = None
    total_latency_ms: float | None = None
    audio_duration_ms: float | None = None
    rtf: float | None = None
    server_ttfa_ms: float | None = None
    error_code: str | None = None
    error: str | None = None
    audio_format: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    audio_payload: bytes | None = field(default=None, repr=False)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        item = asdict(self)
        item.pop("audio_payload", None)
        return item

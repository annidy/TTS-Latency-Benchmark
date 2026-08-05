from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio
import json
from typing import Any
from urllib.parse import urlencode

from ....models import AudioFormat, EventType, ProviderEvent, SynthesisRequest
from ...base import AdapterProtocol, MultiTransportModelAdapter
from ...transports.websocket import WebSocketTransportBase


class BreezeBlueBase(MultiTransportModelAdapter):
    http_protocol = AdapterProtocol(
        name="HTTP",
        framing="audio bytes in HTTP response body",
        first_response_boundary="HTTP response headers available",
        completion_boundary="HTTP response body EOF",
        connection_reuse="persistent HTTP connection pool",
    )
    websocket_protocol = AdapterProtocol(
        name="WebSocket",
        framing="JSON control events and binary PCM frames",
        first_response_boundary="first application-visible turn message",
        completion_boundary="turn.done event",
        connection_reuse="multiple turns per connection",
    )

    def __init__(self, provider_id: str, config: dict[str, Any], clock: Any) -> None:
        super().__init__(provider_id, config, clock)
        self._session_ready = False
        self._max_append_bytes = 2048

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[str]:
        return cls.missing(config, "endpoint.url", "credentials.api_key", "voice.value")

    def _auth_headers(self) -> dict[str, str]:
        if str(self.setting("request.auth_header", "xi-api-key")).lower() == "xi-api-key":
            return {"xi-api-key": self.api_key}
        return {"Authorization": f"Bearer {self.api_key}"}

    async def synthesize_http(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        payload: dict[str, Any] = {"text": request.text, "model_id": self.model_id}
        if self.setting("request.instructions"):
            payload["instructions"] = self.setting("request.instructions")
        if self.setting("request.voice_settings") is not None:
            payload["voice_settings"] = self.setting("request.voice_settings")
        encoding = str(self.setting("audio.encoding", "mp3"))
        sample_rate = int(self.setting("audio.sample_rate_hz", 44_100))
        async for event in self.stream_http_bytes(
            url=self.endpoint_url(voice_id=request.voice or self.voice_value),
            headers={**self._auth_headers(), "Content-Type": "application/json"},
            payload=payload,
            audio_format=AudioFormat(encoding, sample_rate),
            error_prefix="BreezeBlue",
        ):
            yield event

    def websocket_url(self) -> str:
        query: dict[str, Any] = {
            "model_id": self.model_id,
            "enable_logging": str(bool(self.setting("request.enable_logging", False))).lower(),
        }
        if self.setting("request.language_code"):
            query["language_code"] = self.setting("request.language_code")
        if self.setting("request.inactivity_timeout_seconds") is not None:
            query["inactivity_timeout_seconds"] = int(
                self.setting("request.inactivity_timeout_seconds")
            )
        return f"{self.endpoint_url(voice_id=self.voice_value)}?{urlencode(query)}"

    def websocket_headers(self) -> dict[str, str]:
        return self._auth_headers()

    async def _ensure_ready(self, socket: Any) -> ProviderEvent | None:
        if self._session_ready:
            return None
        while True:
            data = await self.receive_json(socket)
            now = self.clock.now_ns()
            if data.get("type") == "error":
                raise RuntimeError(f"BreezeBlue WebSocket error: {data}")
            if data.get("type") == "session.ready":
                self._session_ready = True
                self._max_append_bytes = int(data.get("max_append_bytes", 2048))
                return ProviderEvent(
                    EventType.FIRST_RESPONSE_BYTE,
                    now,
                    data={"server_event": "session.ready", "session_id": data.get("session_id")},
                )

    async def synthesize_websocket(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        socket = await self.connect_websocket()
        ready = await self._ensure_ready(socket)
        if ready:
            yield ready
        turn_id = f"ttfa-{request.request_id}"
        await socket.send(json.dumps({"type": "turn.start", "turn_id": turn_id}))
        for part in split_utf8(request.text, self._max_append_bytes):
            await socket.send(json.dumps({"type": "text.append", "text": part}))
        await socket.send(json.dumps({"type": "text.flush"}))
        await socket.send(json.dumps({"type": "turn.end"}))
        yield ProviderEvent(EventType.REQUEST_BODY_SENT, self.clock.now_ns())
        first = ready is not None
        server_timing = False
        sequence = 0
        fmt = AudioFormat(
            str(self.setting("audio.encoding", "pcm_s16le")),
            int(self.setting("audio.sample_rate_hz", 24_000)),
        )
        while True:
            raw = await asyncio.wait_for(
                socket.recv(), timeout=float(self.setting("request.timeout_s", 60))
            )
            now = self.clock.now_ns()
            if isinstance(raw, bytes):
                if not first:
                    first = True
                    yield ProviderEvent(EventType.FIRST_RESPONSE_BYTE, now)
                yield ProviderEvent(
                    EventType.AUDIO_CHUNK,
                    now,
                    data={"sequence": sequence, "turn_id": turn_id},
                    audio=raw,
                    audio_format=fmt,
                )
                sequence += 1
                continue
            data = json.loads(raw)
            event_type = data.get("type")
            if event_type in {"usage.committed", "pong"}:
                continue
            if data.get("turn_id") not in {None, turn_id}:
                continue
            if not first:
                first = True
                yield ProviderEvent(
                    EventType.FIRST_RESPONSE_BYTE, now, data={"server_event": event_type}
                )
            if event_type == "error":
                raise RuntimeError(f"BreezeBlue WebSocket error: {data}")
            if event_type == "audio.started" and data.get("ttfa_ms") is not None:
                server_timing = True
                yield ProviderEvent(
                    EventType.SERVER_TIMING,
                    now,
                    data={"ttfa_ms": float(data["ttfa_ms"]), "turn_id": turn_id},
                )
            if event_type == "turn.done":
                if not server_timing and data.get("ttfa_ms") is not None:
                    yield ProviderEvent(
                        EventType.SERVER_TIMING,
                        now,
                        data={"ttfa_ms": float(data["ttfa_ms"]), "turn_id": turn_id},
                    )
                yield ProviderEvent(EventType.STREAM_FINISHED, now, data={"turn_id": turn_id})
                break

    async def close_websocket(self) -> None:
        if self._socket is not None:
            try:
                await self._socket.send(json.dumps({"type": "session.close"}))
            except Exception:
                pass
        await WebSocketTransportBase.close_websocket(self)
        self._session_ready = False


def split_utf8(text: str, max_bytes: int) -> list[str]:
    if max_bytes <= 0:
        raise ValueError("max_append_bytes must be positive")
    parts: list[str] = []
    current = ""
    for character in text:
        if current and len((current + character).encode()) > max_bytes:
            parts.append(current)
            current = ""
        current += character
    if current:
        parts.append(current)
    return parts

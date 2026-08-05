from __future__ import annotations

from collections.abc import AsyncIterator
import base64
import json
from typing import Any
import uuid

from ....models import AudioFormat, EventType, ProviderEvent, SynthesisRequest
from ...base import AdapterProtocol, MultiTransportModelAdapter


class CartesiaBase(MultiTransportModelAdapter):
    http_protocol = AdapterProtocol(
        name="HTTP",
        framing="raw PCM bytes in HTTP response body",
        first_response_boundary="HTTP response headers available",
        completion_boundary="HTTP response body EOF",
        connection_reuse="persistent HTTP connection pool",
    )
    websocket_protocol = AdapterProtocol(
        name="WebSocket",
        framing="JSON messages with base64 PCM audio",
        first_response_boundary="first message for the context",
        completion_boundary="done=true message for the context",
        connection_reuse="multiple independent contexts per connection",
    )

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[str]:
        return cls.missing(config, "endpoint.url", "credentials.api_key", "voice.value")

    def _headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "Cartesia-Version": str(self.setting("request.api_version", "2026-03-01")),
        }

    def _output_format(self) -> dict[str, Any]:
        return {
            "container": self.setting("audio.container", "raw"),
            "encoding": self.setting("audio.encoding", "pcm_s16le"),
            "sample_rate": int(self.setting("audio.sample_rate_hz", 24_000)),
        }

    async def synthesize_http(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        payload = {
            "model_id": self.model_id,
            "transcript": request.text,
            "voice": {"mode": "id", "id": request.voice or self.voice_value},
            "language": request.language or self.setting("request.language", "en"),
            "output_format": self._output_format(),
        }
        async for event in self.stream_http_bytes(
            url=self.endpoint_url(),
            headers={**self._headers(), "Content-Type": "application/json"},
            payload=payload,
            audio_format=AudioFormat(
                str(self.setting("audio.encoding", "pcm_s16le")),
                int(self.setting("audio.sample_rate_hz", 24_000)),
            ),
            error_prefix="Cartesia",
        ):
            yield event

    def websocket_headers(self) -> dict[str, str]:
        return self._headers()

    async def synthesize_websocket(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        socket = await self.connect_websocket()
        context_id = f"ttfa-{uuid.uuid4()}"
        await socket.send(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "transcript": request.text,
                    "voice": {"mode": "id", "id": request.voice or self.voice_value},
                    "language": request.language or self.setting("request.language", "en"),
                    "context_id": context_id,
                    "output_format": self._output_format(),
                    "add_timestamps": False,
                    "continue": False,
                }
            )
        )
        yield ProviderEvent(EventType.REQUEST_BODY_SENT, self.clock.now_ns())
        first = True
        sequence = 0
        fmt = AudioFormat(
            str(self.setting("audio.encoding", "pcm_s16le")),
            int(self.setting("audio.sample_rate_hz", 24_000)),
        )
        while True:
            data = await self.receive_json(socket)
            if data.get("context_id") not in {None, context_id}:
                continue
            now = self.clock.now_ns()
            if first:
                first = False
                yield ProviderEvent(EventType.FIRST_RESPONSE_BYTE, now)
            if data.get("type") == "error":
                raise RuntimeError(f"Cartesia WebSocket error: {data}")
            if data.get("type") == "chunk" and data.get("data"):
                yield ProviderEvent(
                    EventType.AUDIO_CHUNK,
                    now,
                    data={"sequence": sequence},
                    audio=base64.b64decode(data["data"]),
                    audio_format=fmt,
                )
                sequence += 1
            if data.get("type") == "done" and data.get("done") is True:
                break
        yield ProviderEvent(EventType.STREAM_FINISHED, self.clock.now_ns())

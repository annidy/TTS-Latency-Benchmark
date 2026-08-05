from __future__ import annotations

from collections.abc import AsyncIterator
import base64
import json
from typing import Any
from urllib.parse import urlencode

from ....models import AudioFormat, EventType, ProviderEvent, SynthesisRequest
from ...base import AdapterProtocol, MultiTransportModelAdapter


class XAIBase(MultiTransportModelAdapter):
    http_protocol = AdapterProtocol(
        name="HTTP",
        framing="provider audio bytes in server-streamed HTTP response",
        first_response_boundary="HTTP response headers available",
        completion_boundary="HTTP response body EOF",
        connection_reuse="persistent HTTP connection pool",
    )
    websocket_protocol = AdapterProtocol(
        name="WebSocket",
        framing="JSON events with base64 audio.delta payloads",
        first_response_boundary="first event for the utterance",
        completion_boundary="audio.done event",
        connection_reuse="multiple utterances per connection",
    )

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[str]:
        return cls.missing(config, "endpoint.url", "credentials.api_key", "voice.value")

    async def synthesize_http(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        payload = {
            "text": request.text,
            "voice_id": request.voice or self.voice_value,
            "language": request.language or self.setting("request.language", "en"),
        }
        async for event in self.stream_http_bytes(
            url=self.endpoint_url(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            audio_format=AudioFormat(
                str(self.setting("audio.encoding", "mp3")),
                int(self.setting("audio.sample_rate_hz", 24_000)),
            ),
            error_prefix="xAI TTS",
        ):
            yield event

    def websocket_url(self) -> str:
        query = {
            "language": self.setting("request.language", "en"),
            "voice": self.voice_value,
            "codec": "pcm",
            "sample_rate": int(self.setting("audio.sample_rate_hz", 24_000)),
            "optimize_streaming_latency": int(
                self.setting("request.optimize_streaming_latency", 1)
            ),
            "text_normalization": str(
                bool(self.setting("request.text_normalization", False))
            ).lower(),
        }
        return f"{self.endpoint_url()}?{urlencode(query)}"

    def websocket_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def synthesize_websocket(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        socket = await self.connect_websocket()
        await socket.send(json.dumps({"type": "text.delta", "delta": request.text}))
        await socket.send(json.dumps({"type": "text.done"}))
        yield ProviderEvent(EventType.REQUEST_BODY_SENT, self.clock.now_ns())
        first = True
        sequence = 0
        fmt = AudioFormat("pcm_s16le", int(self.setting("audio.sample_rate_hz", 24_000)))
        while True:
            data = await self.receive_json(socket)
            now = self.clock.now_ns()
            if first:
                first = False
                yield ProviderEvent(EventType.FIRST_RESPONSE_BYTE, now)
            if data.get("type") == "error":
                raise RuntimeError(f"xAI TTS WebSocket error: {data}")
            if data.get("type") == "audio.delta" and data.get("delta"):
                yield ProviderEvent(
                    EventType.AUDIO_CHUNK,
                    now,
                    data={"sequence": sequence},
                    audio=base64.b64decode(data["delta"]),
                    audio_format=fmt,
                )
                sequence += 1
            if data.get("type") == "audio.done":
                break
        yield ProviderEvent(EventType.STREAM_FINISHED, self.clock.now_ns())

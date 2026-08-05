from __future__ import annotations

from collections.abc import AsyncIterator
import base64
import json
from typing import Any
from urllib.parse import urlencode

from ....models import AudioFormat, EventType, ProviderEvent, SynthesisRequest
from ...base import AdapterProtocol, MultiTransportModelAdapter


class ElevenLabsBase(MultiTransportModelAdapter):
    """Authentication and wire protocols shared by supported ElevenLabs models."""

    http_protocol = AdapterProtocol(
        name="HTTP",
        framing="raw PCM bytes in chunked HTTP response",
        first_response_boundary="HTTP response headers available",
        completion_boundary="HTTP response body EOF",
        connection_reuse="persistent HTTP connection pool",
    )
    websocket_protocol = AdapterProtocol(
        name="WebSocket",
        framing="JSON messages with base64 PCM audio",
        first_response_boundary="first message for the request context",
        completion_boundary="isFinal marker for the request context",
        connection_reuse="multiple independent contexts per connection",
    )

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[str]:
        return cls.missing(config, "endpoint.url", "credentials.api_key", "voice.value")

    async def synthesize_http(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        output = str(self.setting("audio.output_format", "pcm_24000"))
        payload: dict[str, Any] = {"text": request.text, "model_id": self.model_id}
        for source, target in (
            ("request.language_code", "language_code"),
            ("request.text_normalization", "apply_text_normalization"),
            ("request.voice_settings", "voice_settings"),
            ("request.seed", "seed"),
        ):
            if self.setting(source) is not None:
                payload[target] = self.setting(source)
        fmt = AudioFormat(
            "pcm_s16le" if output.startswith("pcm_") else "mp3",
            int(output.split("_")[1]) if output.startswith("pcm_") else 44_100,
        )
        async for event in self.stream_http_bytes(
            url=self.endpoint_url(voice_id=request.voice or self.voice_value),
            query={"output_format": output},
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            payload=payload,
            audio_format=fmt,
            error_prefix="ElevenLabs",
        ):
            yield event

    def websocket_url(self) -> str:
        output = str(self.setting("audio.output_format", "pcm_24000"))
        query: dict[str, Any] = {
            "model_id": self.model_id,
            "output_format": output,
            "auto_mode": str(bool(self.setting("request.auto_mode", True))).lower(),
            "sync_alignment": str(bool(self.setting("request.sync_alignment", False))).lower(),
        }
        if self.setting("request.text_normalization"):
            query["apply_text_normalization"] = self.setting("request.text_normalization")
        if self.setting("request.language_code"):
            query["language_code"] = self.setting("request.language_code")
        return f"{self.endpoint_url(voice_id=self.voice_value)}?{urlencode(query)}"

    def websocket_headers(self) -> dict[str, str]:
        return {"xi-api-key": self.api_key}

    async def synthesize_websocket(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        context_id = f"ttfa-{request.request_id}"
        socket = await self.connect_websocket()
        initialize: dict[str, Any] = {"text": " ", "context_id": context_id}
        if self.setting("request.voice_settings") is not None:
            initialize["voice_settings"] = self.setting("request.voice_settings")
        await socket.send(json.dumps(initialize))
        await socket.send(json.dumps({"text": request.text + " ", "context_id": context_id}))
        await socket.send(json.dumps({"context_id": context_id, "flush": True}))
        await socket.send(json.dumps({"context_id": context_id, "close_context": True}))
        yield ProviderEvent(EventType.REQUEST_BODY_SENT, self.clock.now_ns())
        first = True
        sequence = 0
        output = str(self.setting("audio.output_format", "pcm_24000"))
        fmt = AudioFormat(
            "pcm_s16le" if output.startswith("pcm_") else "mp3",
            int(output.split("_")[1]) if output.startswith("pcm_") else 44_100,
        )
        while True:
            data = await self.receive_json(socket)
            if (data.get("contextId") or data.get("context_id")) not in {None, context_id}:
                continue
            now = self.clock.now_ns()
            if first:
                first = False
                yield ProviderEvent(EventType.FIRST_RESPONSE_BYTE, now)
            if data.get("error"):
                raise RuntimeError(f"ElevenLabs WebSocket error: {data['error']}")
            if data.get("audio"):
                yield ProviderEvent(
                    EventType.AUDIO_CHUNK,
                    now,
                    data={"sequence": sequence},
                    audio=base64.b64decode(data["audio"]),
                    audio_format=fmt,
                )
                sequence += 1
            if data.get("isFinal") is True or data.get("is_final") is True:
                break
        yield ProviderEvent(EventType.STREAM_FINISHED, self.clock.now_ns())

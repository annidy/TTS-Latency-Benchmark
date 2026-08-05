from __future__ import annotations

from collections.abc import AsyncIterator
import base64
import json
from typing import Any
import uuid

from ....models import AudioFormat, EventType, ProviderEvent, SynthesisRequest
from ...base import AdapterProtocol, MultiTransportModelAdapter


class InworldBase(MultiTransportModelAdapter):
    http_protocol = AdapterProtocol(
        name="HTTP",
        framing="JSON response containing base64 WAV audio",
        first_response_boundary="HTTP response headers available",
        completion_boundary="complete JSON response decoded",
        connection_reuse="persistent HTTP connection pool",
    )
    websocket_protocol = AdapterProtocol(
        name="WebSocket",
        framing="JSON messages with base64 PCM audio",
        first_response_boundary="first application-visible WebSocket message",
        completion_boundary="contextClosed response",
        connection_reuse="multiple independent contexts per connection",
    )

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[str]:
        return cls.missing(config, "endpoint.url", "credentials.api_key", "voice.value")

    def _audio_config(self) -> dict[str, Any]:
        return {
            "audioEncoding": self.setting("audio.wire_encoding", "LINEAR16"),
            "sampleRateHertz": int(self.setting("audio.sample_rate_hz", 24_000)),
        }

    async def synthesize_http(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        payload: dict[str, Any] = {
            "text": request.text,
            "voiceId": request.voice or self.voice_value,
            "modelId": self.model_id,
            "audioConfig": self._audio_config(),
            "deliveryMode": self.setting("request.delivery_mode", "BALANCED"),
            "applyTextNormalization": self.setting("request.text_normalization", "OFF"),
        }
        if request.language or self.setting("request.language"):
            payload["language"] = request.language or self.setting("request.language")
        client = await self.http_client()
        yield ProviderEvent(EventType.REQUEST_BODY_SENT, self.clock.now_ns())
        async with client.stream(
            "POST",
            self.endpoint_url(),
            headers={
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            yield ProviderEvent(
                EventType.FIRST_RESPONSE_BYTE,
                self.clock.now_ns(),
                data={"http_status": response.status_code},
            )
            body = await response.aread()
            if response.is_error:
                raise RuntimeError(
                    f"Inworld HTTP {response.status_code}: " + body.decode(errors="replace")[:500]
                )
            data = json.loads(body)
            audio = data.get("audioContent")
            if not audio:
                raise RuntimeError(f"Inworld HTTP response did not contain audioContent: {data}")
            yield ProviderEvent(
                EventType.AUDIO_CHUNK,
                self.clock.now_ns(),
                data={"sequence": 0},
                audio=base64.b64decode(audio),
                audio_format=AudioFormat(
                    str(self.setting("audio.encoding", "wav")),
                    int(self.setting("audio.sample_rate_hz", 24_000)),
                ),
            )
        yield ProviderEvent(EventType.STREAM_FINISHED, self.clock.now_ns())

    def websocket_headers(self) -> dict[str, str]:
        return {"Authorization": f"Basic {self.api_key}"}

    async def synthesize_websocket(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        socket = await self.connect_websocket()
        context_id = f"ttfa-{uuid.uuid4()}"
        sample_rate = int(self.setting("audio.sample_rate_hz", 24_000))
        wire_encoding = str(self.setting("audio.wire_encoding", "PCM"))
        create: dict[str, Any] = {
            "voiceId": request.voice or self.voice_value,
            "modelId": self.model_id,
            "deliveryMode": self.setting("request.delivery_mode", "BALANCED"),
            "autoMode": bool(self.setting("request.auto_mode", True)),
            "timestampType": self.setting("request.timestamp_type", "TIMESTAMP_TYPE_UNSPECIFIED"),
            "timestampTransportStrategy": self.setting(
                "request.timestamp_transport_strategy", "ASYNC"
            ),
            "audioConfig": {
                "audioEncoding": wire_encoding,
                "sampleRateHertz": sample_rate,
            },
        }
        if request.language or self.setting("request.language"):
            create["language"] = request.language or self.setting("request.language")
        if self.setting("request.text_normalization"):
            create["applyTextNormalization"] = self.setting("request.text_normalization")
        await socket.send(json.dumps({"create": create, "contextId": context_id}))
        yield ProviderEvent(EventType.REQUEST_BODY_SENT, self.clock.now_ns())
        first = True
        text_sent = False
        close_requested = False
        sequence = 0
        fmt = AudioFormat("pcm_s16le", sample_rate)
        while True:
            data = await self.receive_json(socket)
            now = self.clock.now_ns()
            result = data.get("result", {})
            if first:
                first = False
                yield ProviderEvent(
                    EventType.FIRST_RESPONSE_BYTE,
                    now,
                    data={
                        "server_event": next(
                            (key for key in result if key not in {"contextId", "status"}),
                            None,
                        ),
                        "context_config": result.get("contextCreated"),
                    },
                )
            status = result.get("status")
            if status and int(status.get("code", 0)) != 0:
                raise RuntimeError(f"Inworld WebSocket error: {status}")
            if "contextCreated" in result and not text_sent:
                await socket.send(
                    json.dumps(
                        {
                            "send_text": {"text": request.text, "flush_context": {}},
                            "contextId": context_id,
                        }
                    )
                )
                text_sent = True
            audio = result.get("audioChunk", {}).get("audioContent")
            if audio:
                yield ProviderEvent(
                    EventType.AUDIO_CHUNK,
                    now,
                    data={"sequence": sequence},
                    audio=base64.b64decode(audio),
                    audio_format=fmt,
                )
                sequence += 1
            if "flushCompleted" in result and not close_requested:
                await socket.send(json.dumps({"close_context": {}, "contextId": context_id}))
                close_requested = True
            if "contextClosed" in result:
                break
        yield ProviderEvent(EventType.STREAM_FINISHED, self.clock.now_ns())

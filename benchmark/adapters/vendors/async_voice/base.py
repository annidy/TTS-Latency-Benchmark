from __future__ import annotations

from collections.abc import AsyncIterator
import base64
import json
from typing import Any
from urllib.parse import urlencode
import uuid

from ....models import AudioFormat, EventType, ProviderEvent, SynthesisRequest
from ...base import AdapterProtocol, MultiTransportModelAdapter
from ...transports.websocket import WebSocketTransportBase


class AsyncVoiceBase(MultiTransportModelAdapter):
    http_protocol = AdapterProtocol(
        name="HTTP",
        framing="raw float32 PCM bytes in chunked HTTP response",
        first_response_boundary="HTTP response headers available",
        completion_boundary="HTTP response body EOF",
        connection_reuse="persistent HTTP connection pool",
    )
    websocket_protocol = AdapterProtocol(
        name="WebSocket",
        framing="JSON messages with base64 float32 PCM audio",
        first_response_boundary="first response for the context",
        completion_boundary="final=true for the context",
        connection_reuse="one initialized connection with a context per synthesis turn",
    )

    def __init__(self, provider_id: str, config: dict[str, Any], clock: Any) -> None:
        super().__init__(provider_id, config, clock)
        self._async_initialized = False

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[str]:
        return cls.missing(config, "endpoint.url", "credentials.api_key", "voice.value")

    def _output_format(self) -> dict[str, Any]:
        return {
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": int(self.setting("audio.sample_rate_hz", 44_100)),
        }

    async def synthesize_http(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        payload = {
            "model_id": self.model_id,
            "transcript": request.text,
            "voice": {"mode": "id", "id": request.voice or self.voice_value},
            "output_format": self._output_format(),
        }
        async for event in self.stream_http_bytes(
            url=self.endpoint_url(),
            headers={
                "x-api-key": self.api_key,
                "version": "v1",
                "Content-Type": "application/json",
            },
            payload=payload,
            audio_format=AudioFormat(
                "pcm_f32le",
                int(self.setting("audio.sample_rate_hz", 44_100)),
                1,
                4,
            ),
            error_prefix="Async Voice",
        ):
            yield event

    def websocket_url(self) -> str:
        return f"{self.endpoint_url()}?{urlencode({'api_key': self.api_key, 'version': 'v1'})}"

    async def _initialize(self, socket: Any) -> None:
        if self._async_initialized:
            return
        await socket.send(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "voice": {"mode": "id", "id": self.voice_value},
                    "output_format": self._output_format(),
                }
            )
        )
        acknowledgement = await self.receive_json(socket)
        if acknowledgement.get("error_code"):
            raise RuntimeError(f"Async Voice initialization error: {acknowledgement}")
        if acknowledgement.get("event") != "init_ack":
            raise RuntimeError(f"Async Voice did not acknowledge initialization: {acknowledgement}")
        self._async_initialized = True

    async def synthesize_websocket(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        socket = await self.connect_websocket()
        await self._initialize(socket)
        context_id = str(uuid.uuid4())
        await socket.send(
            json.dumps(
                {
                    "context_id": context_id,
                    "transcript": request.text.rstrip() + " ",
                    "force": True,
                }
            )
        )
        yield ProviderEvent(EventType.REQUEST_BODY_SENT, self.clock.now_ns())
        first = True
        sequence = 0
        close_requested = False
        fmt = AudioFormat("pcm_f32le", int(self.setting("audio.sample_rate_hz", 44_100)), 1, 4)
        while True:
            data = await self.receive_json(socket)
            if data.get("context_id") not in {None, context_id}:
                continue
            now = self.clock.now_ns()
            if first:
                first = False
                yield ProviderEvent(EventType.FIRST_RESPONSE_BYTE, now)
            if data.get("error_code"):
                raise RuntimeError(f"Async Voice WebSocket error: {data}")
            if data.get("audio"):
                yield ProviderEvent(
                    EventType.AUDIO_CHUNK,
                    now,
                    data={"sequence": sequence},
                    audio=base64.b64decode(data["audio"]),
                    audio_format=fmt,
                )
                sequence += 1
                if not close_requested:
                    await socket.send(
                        json.dumps(
                            {
                                "context_id": context_id,
                                "close_context": True,
                                "transcript": "",
                            }
                        )
                    )
                    close_requested = True
            if data.get("final") is True:
                break
        yield ProviderEvent(EventType.STREAM_FINISHED, self.clock.now_ns())

    async def close_websocket(self) -> None:
        if self._socket is not None:
            try:
                await self._socket.send(json.dumps({"terminate": True}))
            except Exception:
                pass
        await WebSocketTransportBase.close_websocket(self)
        self._async_initialized = False

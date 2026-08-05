from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio
from typing import Any

import msgpack

from ....models import AudioFormat, EventType, ProviderEvent, SynthesisRequest
from ...base import AdapterProtocol, MultiTransportModelAdapter
from ...transports.websocket import WebSocketTransportBase


class FishAudioBase(MultiTransportModelAdapter):
    http_protocol = AdapterProtocol(
        name="HTTP",
        framing="raw audio bytes in chunked HTTP response",
        first_response_boundary="HTTP response headers available",
        completion_boundary="HTTP response body EOF",
        connection_reuse="persistent HTTP connection pool",
    )
    websocket_protocol = AdapterProtocol(
        name="WebSocket",
        framing="MessagePack events with binary audio",
        first_response_boundary="first MessagePack server event",
        completion_boundary="configured audio-idle drain after each flush",
        connection_reuse="one start event and multiple serialized text/flush turns",
    )

    def __init__(self, provider_id: str, config: dict[str, Any], clock: Any) -> None:
        super().__init__(provider_id, config, clock)
        self._fish_started = False

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[str]:
        return cls.missing(config, "endpoint.url", "credentials.api_key", "voice.value")

    async def synthesize_http(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        output = str(self.setting("audio.format", "pcm"))
        sample_rate = int(self.setting("audio.sample_rate_hz", 24_000))
        payload = {
            "text": request.text,
            "reference_id": request.voice or self.voice_value,
            "format": output,
            "sample_rate": sample_rate,
            "chunk_length": int(self.setting("request.chunk_length", 100)),
            "latency": self.setting("request.latency", "low"),
        }
        async for event in self.stream_http_bytes(
            url=self.endpoint_url(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "model": self.model_id,
            },
            payload=payload,
            audio_format=AudioFormat("pcm_s16le" if output == "pcm" else output, sample_rate),
            error_prefix="Fish Audio",
        ):
            yield event

    def websocket_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "model": self.model_id}

    async def _ensure_started(self, socket: Any, request: SynthesisRequest) -> None:
        if self._fish_started:
            return
        start = {
            "text": "",
            "format": self.setting("audio.format", "pcm"),
            "sample_rate": int(self.setting("audio.sample_rate_hz", 24_000)),
            "chunk_length": int(self.setting("request.chunk_length", 100)),
            "reference_id": request.voice or self.voice_value,
            "latency": self.setting("request.latency", "low"),
            "condition_on_previous_chunks": bool(
                self.setting("request.condition_on_previous_chunks", True)
            ),
        }
        await socket.send(msgpack.packb({"event": "start", "request": start}))
        self._fish_started = True

    async def synthesize_websocket(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        socket = await self.connect_websocket()
        await self._ensure_started(socket, request)
        output = str(self.setting("audio.format", "pcm"))
        sample_rate = int(self.setting("audio.sample_rate_hz", 24_000))
        await socket.send(msgpack.packb({"event": "text", "text": request.text}))
        await socket.send(msgpack.packb({"event": "flush"}))
        yield ProviderEvent(EventType.REQUEST_BODY_SENT, self.clock.now_ns())
        first = True
        sequence = 0
        received_audio = False
        while True:
            timeout = float(
                self.setting(
                    "request.turn_idle_timeout_s" if received_audio else "request.timeout_s",
                    1.5 if received_audio else 60,
                )
            )
            try:
                raw = await asyncio.wait_for(socket.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                if received_audio:
                    break
                raise RuntimeError("Fish Audio WebSocket timed out before first audio")
            now = self.clock.now_ns()
            if first:
                first = False
                yield ProviderEvent(EventType.FIRST_RESPONSE_BYTE, now)
            if isinstance(raw, str):
                raw = raw.encode("latin1")
            data = msgpack.unpackb(raw, raw=False)
            if data.get("event") == "audio" and data.get("audio"):
                received_audio = True
                yield ProviderEvent(
                    EventType.AUDIO_CHUNK,
                    now,
                    data={"sequence": sequence},
                    audio=bytes(data["audio"]),
                    audio_format=AudioFormat(
                        "pcm_s16le" if output == "pcm" else output, sample_rate
                    ),
                )
                sequence += 1
            if data.get("event") == "finish":
                raise RuntimeError(f"Fish Audio session ended before adapter close: {data}")
        yield ProviderEvent(EventType.STREAM_FINISHED, self.clock.now_ns())

    async def close_websocket(self) -> None:
        socket = self._socket
        if socket is not None and self._fish_started:
            try:
                await socket.send(msgpack.packb({"event": "stop"}))
                while True:
                    raw = await asyncio.wait_for(
                        socket.recv(), timeout=float(self.setting("request.timeout_s", 60))
                    )
                    if isinstance(raw, str):
                        raw = raw.encode("latin1")
                    data = msgpack.unpackb(raw, raw=False)
                    if data.get("event") == "finish":
                        if data.get("reason") == "error":
                            raise RuntimeError(f"Fish Audio WebSocket error: {data}")
                        break
            finally:
                await WebSocketTransportBase.close_websocket(self)
                self._fish_started = False
            return
        await WebSocketTransportBase.close_websocket(self)
        self._fish_started = False

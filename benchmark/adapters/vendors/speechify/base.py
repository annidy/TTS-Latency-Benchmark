from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ....models import AudioFormat, ProviderEvent, SynthesisRequest
from ...base import AdapterProtocol, MultiTransportModelAdapter


class SpeechifyBase(MultiTransportModelAdapter):
    http_protocol = AdapterProtocol(
        name="HTTP",
        framing="raw little-endian PCM in chunked HTTP response",
        first_response_boundary="HTTP response headers available",
        completion_boundary="HTTP response body EOF",
        connection_reuse="persistent HTTP connection pool",
    )

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[str]:
        return cls.missing(config, "endpoint.url", "credentials.api_key", "voice.value")

    async def synthesize_http(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        sample_rate = int(self.setting("audio.sample_rate_hz", 24_000))
        payload = {
            "input": request.text,
            "voice_id": request.voice or self.voice_value,
            "model": self.model_id,
            "language": request.language or self.setting("request.language", "en-US"),
            "options": {"output_format": f"pcm_{sample_rate}"},
        }
        async for event in self.stream_http_bytes(
            url=self.endpoint_url(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/pcm",
                "Speechify-Version": str(self.setting("request.api_version", "2026-06-25")),
            },
            payload=payload,
            audio_format=AudioFormat("pcm_s16le", sample_rate),
            error_prefix="Speechify",
        ):
            yield event

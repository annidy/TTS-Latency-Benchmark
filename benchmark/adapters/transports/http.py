from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...models import AudioFormat, EventType, ProviderEvent


class HttpTransportBase:
    """Reusable HTTP/2 connection-pool and raw streaming event conversion."""

    _http_client: httpx.AsyncClient | None = None

    async def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                http2=True,
                timeout=float(self.setting("request.timeout_s", 60)),
            )
        return self._http_client

    async def prepare_http(self) -> None:
        await self.http_client()

    async def close_http(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def stream_http_bytes(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        audio_format: AudioFormat,
        query: dict[str, Any] | None = None,
        error_prefix: str,
    ) -> AsyncIterator[ProviderEvent]:
        client = await self.http_client()
        yield ProviderEvent(EventType.REQUEST_BODY_SENT, self.clock.now_ns())
        async with client.stream(
            "POST", url, params=query, headers=headers, json=payload
        ) as response:
            request_id = (
                response.headers.get("request-id")
                or response.headers.get("x-request-id")
                or response.headers.get("x-trace-id")
            )
            yield ProviderEvent(
                EventType.FIRST_RESPONSE_BYTE,
                self.clock.now_ns(),
                data={"http_status": response.status_code, "request_id": request_id},
            )
            if response.is_error:
                detail = (await response.aread()).decode(errors="replace")[:500]
                raise RuntimeError(f"{error_prefix} HTTP {response.status_code}: {detail}")
            sequence = 0
            effective_format: AudioFormat | None = None
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                if effective_format is None:
                    effective_format = _detect_audio_format(
                        audio_format, response.headers.get("content-type", ""), chunk
                    )
                yield ProviderEvent(
                    EventType.AUDIO_CHUNK,
                    self.clock.now_ns(),
                    data={"sequence": sequence, "request_id": request_id},
                    audio=chunk,
                    audio_format=effective_format,
                )
                sequence += 1
        yield ProviderEvent(EventType.STREAM_FINISHED, self.clock.now_ns())


def _detect_audio_format(
    configured: AudioFormat, content_type: str, first_chunk: bytes
) -> AudioFormat:
    """Honor provider response containers even when a request asks for raw PCM."""
    media_type = content_type.lower().split(";", 1)[0].strip()
    if first_chunk.startswith(b"RIFF") or media_type in {"audio/wav", "audio/x-wav"}:
        return AudioFormat("wav", configured.sample_rate_hz, configured.channels)
    if first_chunk.startswith(b"ID3") or media_type in {"audio/mpeg", "audio/mp3"}:
        return AudioFormat("mp3", configured.sample_rate_hz, configured.channels)
    return configured

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..clock import Clock
from ..models import ProviderEvent, SynthesisRequest
from .transports.http import HttpTransportBase
from .transports.websocket import WebSocketTransportBase


@dataclass(frozen=True)
class ModelCapabilities:
    transports: tuple[str, ...]
    websocket_persistent: bool = False
    supports_incremental_text: bool = False
    provides_server_timing: bool = False


@dataclass(frozen=True)
class AdapterProtocol:
    name: str
    framing: str
    first_response_boundary: str
    completion_boundary: str
    connection_reuse: str


class ModelAdapter(ABC):
    """The only provider-facing contract used by the benchmark engine."""

    kind: str
    capabilities: ModelCapabilities

    def __init__(self, provider_id: str, config: dict[str, Any], clock: Clock) -> None:
        self.provider_id = provider_id
        self.config = config
        self.clock = clock

    @classmethod
    def missing(cls, config: dict[str, Any], *paths: str) -> list[str]:
        return [path for path in paths if not _nested(config, path)]

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> list[str]:
        return []

    def setting(self, path: str, default: Any = None) -> Any:
        value = _nested(self.config, path)
        return default if value is None else value

    def endpoint_url(self, **template_values: str) -> str:
        url = str(self.setting("endpoint.url", ""))
        return url.format(**template_values) if template_values else url

    @property
    def api_key(self) -> str:
        return str(self.setting("credentials.api_key", ""))

    @property
    def model_id(self) -> str:
        return str(self.setting("model.id", ""))

    @property
    def voice_value(self) -> str:
        return str(self.setting("voice.value", ""))

    @property
    def transport(self) -> str:
        value = str(self.config.get("transport", ""))
        if value in {"http", "http_stream"}:
            return "http"
        if value == "websocket":
            return "websocket"
        raise ValueError(f"unsupported transport for {self.provider_id}: {value!r}")

    async def warmup(self, request: SynthesisRequest) -> None:
        async for _ in self.synthesize(request):
            pass

    async def prepare(self) -> None:
        return None

    @abstractmethod
    def protocol_for_transport(self) -> AdapterProtocol:
        raise NotImplementedError

    def measurement_protocol(self, profile: dict[str, Any]) -> dict[str, Any]:
        protocol = self.protocol_for_transport()
        mode = str(profile.get("connection_mode", "new"))
        lifecycle = {
            "new": "new connection per measured sample",
            "persistent": "warm persistent connection/pool",
        }.get(mode, mode)
        return {
            "name": protocol.name,
            "framing": protocol.framing,
            "connection_lifecycle": lifecycle,
            "connection_reuse": protocol.connection_reuse,
            "connection_setup_in_timing": mode == "new",
            "request_start_boundary": (
                "before connection setup and synthesis request"
                if mode == "new"
                else "after excluded warmup, before synthesis request"
            ),
            "first_response_boundary": protocol.first_response_boundary,
            "completion_boundary": protocol.completion_boundary,
        }

    @abstractmethod
    def synthesize(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class MultiTransportModelAdapter(
    HttpTransportBase,
    WebSocketTransportBase,
    ModelAdapter,
):
    """Dispatches the stable model contract to its configured transport."""

    http_protocol = AdapterProtocol(
        name="HTTP",
        framing="provider-defined HTTP response body",
        first_response_boundary="HTTP response headers available",
        completion_boundary="HTTP response body EOF",
        connection_reuse="persistent HTTP connection pool",
    )
    websocket_protocol = AdapterProtocol(
        name="WebSocket",
        framing="provider-defined WebSocket messages",
        first_response_boundary="first application-visible WebSocket message",
        completion_boundary="provider turn completion event",
        connection_reuse="multiple turns per connection",
    )

    def protocol_for_transport(self) -> AdapterProtocol:
        return self.http_protocol if self.transport == "http" else self.websocket_protocol

    async def prepare(self) -> None:
        if self.transport == "http":
            await self.prepare_http()
        else:
            await self.prepare_websocket()

    async def synthesize(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        method = self.synthesize_http if self.transport == "http" else self.synthesize_websocket
        async for event in method(request):
            yield event

    async def close(self) -> None:
        await self.close_http()
        await self.close_websocket()

    async def synthesize_http(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        raise NotImplementedError(f"{self.kind} does not implement HTTP")

    async def synthesize_websocket(self, request: SynthesisRequest) -> AsyncIterator[ProviderEvent]:
        raise NotImplementedError(f"{self.kind} does not implement WebSocket")


def _nested(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value

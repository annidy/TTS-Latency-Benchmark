from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets


class WebSocketTransportBase:
    """Reusable persistent WebSocket lifecycle for vendor model adapters."""

    _socket: Any = None

    def websocket_url(self) -> str:
        return self.endpoint_url()

    def websocket_headers(self) -> dict[str, str]:
        return {}

    async def connect_websocket(self) -> Any:
        if self._socket is None:
            self._socket = await websockets.connect(
                self.websocket_url(),
                additional_headers=self.websocket_headers(),
                open_timeout=float(self.setting("request.connect_timeout_s", 15)),
                max_size=None,
            )
        return self._socket

    async def prepare_websocket(self) -> None:
        await self.connect_websocket()

    async def close_websocket(self) -> None:
        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    async def receive_json(self, socket: Any) -> dict[str, Any]:
        raw = await asyncio.wait_for(
            socket.recv(), timeout=float(self.setting("request.timeout_s", 60))
        )
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

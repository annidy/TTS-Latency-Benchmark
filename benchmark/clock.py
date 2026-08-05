from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    def now_ns(self) -> int: ...

    async def sleep_ms(self, delay_ms: float) -> None: ...


class MonotonicClock:
    def now_ns(self) -> int:
        return time.perf_counter_ns()

    async def sleep_ms(self, delay_ms: float) -> None:
        await asyncio.sleep(max(0.0, delay_ms) / 1000.0)


class ManualClock:
    """Deterministic clock for provider and end-to-end contract tests."""

    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        self._now_ns = start_ns

    def now_ns(self) -> int:
        return self._now_ns

    async def sleep_ms(self, delay_ms: float) -> None:
        self._now_ns += round(max(0.0, delay_ms) * 1_000_000)
        await asyncio.sleep(0)

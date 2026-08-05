from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..clock import Clock
from .base import ModelAdapter


class AdapterRegistry:
    def __init__(self) -> None:
        self._models: dict[str, type[ModelAdapter]] = {}

    def decorator(self, model_id: str) -> Callable[[type[ModelAdapter]], type[ModelAdapter]]:
        def register(adapter: type[ModelAdapter]) -> type[ModelAdapter]:
            if model_id in self._models:
                raise ValueError(f"model adapter already registered: {model_id}")
            self._models[model_id] = adapter
            adapter.kind = model_id
            return adapter

        return register

    def create(
        self,
        model_id: str,
        provider_id: str,
        config: dict[str, Any],
        clock: Clock,
    ) -> ModelAdapter:
        try:
            adapter = self._models[model_id]
        except KeyError as error:
            available = ", ".join(self.models())
            raise ValueError(
                f"unsupported model adapter {model_id!r}; supported: {available}"
            ) from error
        transport = str(config.get("transport", "")).replace("http_stream", "http")
        if transport not in adapter.capabilities.transports:
            raise ValueError(f"{model_id} does not support transport {transport!r}")
        problems = adapter.validate_config(config)
        if problems:
            raise ValueError(f"invalid config for {provider_id}: " + "; ".join(problems))
        return adapter(provider_id, config, clock)

    def models(self) -> list[str]:
        return sorted(self._models)

    # CLI compatibility: kinds now intentionally means public model IDs.
    def kinds(self) -> list[str]:
        return self.models()


registry = AdapterRegistry()

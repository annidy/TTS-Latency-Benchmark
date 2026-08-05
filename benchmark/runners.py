from __future__ import annotations

from typing import Any

from .config import LoadedConfig
from .engine import run_local_matrix
from .models import SampleResult


async def run_local_benchmark(
    config: LoadedConfig, run_id: str, seed: int, *, retain_audio: bool = False
) -> list[SampleResult]:
    samples: list[SampleResult] = []
    for index, client in enumerate(config.clients):
        kind = client.get("runner", {}).get("kind", "local")
        client_seed = seed + index * 100_003
        if kind == "local":
            unresolved = sorted(
                marker for provider in config.providers for marker in _unresolved_values(provider)
            )
            if unresolved:
                raise ValueError(
                    f"local client {client['id']} is missing provider environment: "
                    + ", ".join(unresolved)
                )
            samples.extend(
                await run_local_matrix(
                    run_id=run_id,
                    client=client,
                    providers=config.providers,
                    profiles=config.profiles,
                    texts=config.texts,
                    seed=client_seed,
                    retain_audio=retain_audio,
                )
            )
        else:
            raise ValueError(
                f"core benchmark supports only local single-node clients; "
                f"client {client['id']} configured {kind!r}"
            )
    return samples


def _unresolved_values(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            output.update(_unresolved_values(child))
    elif isinstance(value, list):
        for child in value:
            output.update(_unresolved_values(child))
    elif isinstance(value, str) and value.startswith("<UNRESOLVED:"):
        output.add(value.removeprefix("<UNRESOLVED:").removesuffix(">"))
    return output

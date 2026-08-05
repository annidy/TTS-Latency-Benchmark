from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .resources import resolve_config_reference


class CatalogError(ValueError):
    pass


def expand_suite(raw: dict[str, Any], source: Path) -> dict[str, Any]:
    """Resolve a schema-v2 suite into the engine's normalized runtime mapping."""
    if "catalog" not in raw:
        return copy.deepcopy(raw)
    models = _load_catalog(raw["catalog"], source.parent)
    providers = _expand_targets(raw.get("targets"), models)
    texts = _load_texts(raw.get("texts"), source.parent)
    suite = raw.get("suite") or {}
    if not isinstance(suite, dict) or not suite.get("id"):
        raise CatalogError("suite.id is required")
    return {
        "meta": {
            "suite_id": suite["id"],
            "description": suite.get("description", ""),
        },
        "execution": copy.deepcopy(raw.get("execution") or {}),
        "texts": texts,
        "clients": copy.deepcopy(raw.get("clients") or []),
        "providers": providers,
        "outputs": copy.deepcopy(raw.get("outputs") or {"base_dir": "outputs/runs"}),
    }


def _load_catalog(value: Any, base: Path) -> dict[str, dict[str, Any]]:
    paths = value if isinstance(value, list) else [value]
    if not paths or any(not isinstance(item, str) for item in paths):
        raise CatalogError("catalog must be a path or list of paths")
    models: dict[str, dict[str, Any]] = {}
    for item in paths:
        path = resolve_config_reference(item, base)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        declared = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(declared, dict):
            raise CatalogError(f"catalog {path} must contain a models mapping")
        for model_id, definition in declared.items():
            if model_id in models:
                raise CatalogError(f"duplicate catalog model: {model_id}")
            if not isinstance(definition, dict):
                raise CatalogError(f"catalog model {model_id} must be a mapping")
            models[str(model_id)] = copy.deepcopy(definition)
    return models


def _expand_targets(value: Any, models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    if isinstance(value, dict):
        include = value.get("include")
        transport = value.get("transport")
        if not isinstance(include, list) or not isinstance(transport, str):
            raise CatalogError("targets mapping requires include list and transport")
        targets = [{"model": model, "transport": transport} for model in include]
    elif isinstance(value, list):
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("model"), str):
                raise CatalogError("each target requires a model")
            transports = item.get("transports")
            if transports is not None:
                if not isinstance(transports, list) or not transports:
                    raise CatalogError("target.transports must be a non-empty list")
                targets.extend(
                    {"model": item["model"], "transport": transport} for transport in transports
                )
            elif isinstance(item.get("transport"), str):
                targets.append(copy.deepcopy(item))
            else:
                raise CatalogError("target requires transport or transports")
    else:
        raise CatalogError("targets must be a mapping or list")

    providers: list[dict[str, Any]] = []
    for target in targets:
        canonical = target["model"]
        transport = str(target["transport"])
        if transport not in {"http", "websocket"}:
            raise CatalogError(f"unsupported target transport: {transport}")
        if canonical not in models:
            raise CatalogError(f"target references unknown catalog model: {canonical}")
        definition = copy.deepcopy(models[canonical])
        transports = definition.pop("transports", None)
        if not isinstance(transports, dict) or transport not in transports:
            raise CatalogError(f"catalog model {canonical} does not support {transport}")
        provider = _deep_merge(definition, transports[transport])
        provider["id"] = f"{canonical}/{transport}"
        provider["kind"] = canonical
        provider["transport"] = "http_stream" if transport == "http" else "websocket"
        provider.setdefault("model", {})["canonical_id"] = canonical
        if isinstance(target.get("overrides"), dict):
            provider = _deep_merge(provider, target["overrides"])
        providers.append(provider)
    ids = [provider["id"] for provider in providers]
    if len(ids) != len(set(ids)):
        raise CatalogError("suite contains duplicate model/transport targets")
    return providers


def _load_texts(value: Any, base: Path) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return copy.deepcopy(value)
    if not isinstance(value, dict) or not isinstance(value.get("ref"), str):
        raise CatalogError("texts must be a list or contain ref")
    path = resolve_config_reference(value["ref"], base)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    texts = payload.get("texts") if isinstance(payload, dict) else payload
    if not isinstance(texts, list):
        raise CatalogError(f"text file {path} must contain a texts list")
    include = value.get("include")
    if include is not None:
        if not isinstance(include, list) or not all(isinstance(item, str) for item in include):
            raise CatalogError("texts.include must be a list of text IDs")
        by_id = {str(item.get("id")): item for item in texts if isinstance(item, dict)}
        missing = [text_id for text_id in include if text_id not in by_id]
        if missing:
            raise CatalogError(f"texts.include references unknown IDs: {', '.join(missing)}")
        texts = [by_id[text_id] for text_id in include]
    return texts


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output

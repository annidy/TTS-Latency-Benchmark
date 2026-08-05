from __future__ import annotations

from typing import Any


class ConfigError(ValueError):
    pass


def validate_runtime_config(config: dict[str, Any]) -> None:
    required = {"meta", "execution", "texts", "clients", "providers", "outputs"}
    missing = sorted(required - config.keys())
    if missing:
        raise ConfigError("missing top-level sections: " + ", ".join(missing))
    for section in ("texts", "clients", "providers"):
        if not isinstance(config.get(section), list) or not config[section]:
            raise ConfigError(f"{section} must be a non-empty list")
    profiles = config.get("execution", {}).get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ConfigError("execution.profiles must be a non-empty list")
    for section in ("texts", "clients", "providers"):
        ids = [str(item.get("id", "")) for item in config[section]]
        if any(not item for item in ids):
            raise ConfigError(f"every {section} entry requires id")
        if len(ids) != len(set(ids)):
            raise ConfigError(f"duplicate id in {section}")
    for provider in config["providers"]:
        if not provider.get("kind"):
            raise ConfigError(f"provider {provider['id']} requires kind")
        if provider.get("transport") not in {"http_stream", "websocket"}:
            raise ConfigError(f"provider {provider['id']} has invalid transport")
        for section in ("endpoint", "credentials", "model", "voice", "audio", "request"):
            if not isinstance(provider.get(section), dict):
                raise ConfigError(
                    f"provider {provider['id']} requires a mapping section: {section}"
                )
    for profile in profiles:
        if profile.get("connection_mode") not in {"new", "persistent"}:
            raise ConfigError(
                f"profile {profile.get('id')} connection_mode must be new or persistent"
            )

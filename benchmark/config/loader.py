from __future__ import annotations

from dataclasses import dataclass
import copy
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
import yaml

from .catalog import CatalogError, expand_suite
from .resources import resolve_config_reference
from .schema import ConfigError, validate_runtime_config


@dataclass(frozen=True)
class LoadedConfig:
    path: Path | None
    raw: dict[str, Any]
    resolved: dict[str, Any]

    @property
    def run_id_prefix(self) -> str:
        return str(self.resolved.get("meta", {}).get("suite_id", "ttfa"))

    @property
    def providers(self) -> list[dict[str, Any]]:
        return select_provider_items(self.resolved)

    @property
    def clients(self) -> list[dict[str, Any]]:
        return [item for item in self.resolved["clients"] if item.get("enabled", True)]

    @property
    def profiles(self) -> list[dict[str, Any]]:
        return [
            item for item in self.resolved["execution"]["profiles"] if item.get("enabled", True)
        ]

    @property
    def texts(self) -> list[dict[str, Any]]:
        return self.resolved["texts"]

    @property
    def output_base_dir(self) -> Path:
        return Path(self.resolved.get("outputs", {}).get("base_dir", "outputs/runs"))

    def redacted(self) -> dict[str, Any]:
        return _redact(copy.deepcopy(self.resolved))


def load_config(
    path: Path, env_file: Path | None = None, *, allow_missing_env: bool = False
) -> LoadedConfig:
    try:
        path = resolve_config_reference(path, Path.cwd())
    except ValueError as error:
        raise ConfigError(str(error)) from error
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ConfigError("config root must be a mapping")
    try:
        raw = expand_suite(source, path.resolve())
    except (CatalogError, OSError, ValueError, yaml.YAMLError) as error:
        raise ConfigError(str(error)) from error
    env = dict(os.environ)
    if env_file:
        if not env_file.exists():
            raise ConfigError(f"env file not found: {env_file}")
        env = {
            **{key: value for key, value in dotenv_values(env_file).items() if value is not None},
            **env,
        }
    return load_config_mapping(raw, env=env, path=path, allow_missing_env=allow_missing_env)


def load_config_mapping(
    raw: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    path: Path | None = None,
    allow_missing_env: bool = False,
) -> LoadedConfig:
    validate_runtime_config(raw)
    resolved = _resolve_env_refs(
        copy.deepcopy(raw), env or dict(os.environ), allow_missing=allow_missing_env
    )
    validate_runtime_config(resolved)
    return LoadedConfig(path=path, raw=copy.deepcopy(raw), resolved=resolved)


def select_provider_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in config.get("providers", []) if item.get("enabled", True)]


def collect_required_env(raw: dict[str, Any]) -> list[str]:
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("enabled") is False:
                return
            for key, child in value.items():
                if key.endswith("_env") and isinstance(child, str):
                    found.add(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk({**raw, "providers": select_provider_items(raw)})
    return sorted(found)


def missing_required_env(raw: dict[str, Any], env: dict[str, str] | None = None) -> list[str]:
    values = env or dict(os.environ)
    return [key for key in collect_required_env(raw) if not str(values.get(key, "")).strip()]


def _resolve_env_refs(value: Any, env: dict[str, str], *, allow_missing: bool) -> Any:
    if isinstance(value, dict):
        if value.get("enabled") is False:
            return value
        output: dict[str, Any] = {}
        for key, child in value.items():
            if key.endswith("_env"):
                if not isinstance(child, str) or not child.strip():
                    raise ConfigError(f"{key} must name a non-empty environment variable")
                if not str(env.get(child, "")).strip():
                    if not allow_missing:
                        raise ConfigError(f"required environment variable is missing: {child}")
                    output[key[:-4]] = f"<UNRESOLVED:{child}>"
                else:
                    output[key[:-4]] = env[child]
            else:
                output[key] = _resolve_env_refs(child, env, allow_missing=allow_missing)
        return output
    if isinstance(value, list):
        return [_resolve_env_refs(child, env, allow_missing=allow_missing) for child in value]
    return value


def _redact(value: Any) -> Any:
    secret_words = ("api_key", "authorization", "token", "password", "secret")
    if isinstance(value, dict):
        return {
            key: (
                "***REDACTED***"
                if any(word in key.lower() for word in secret_words)
                else _redact(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact(child) for child in value]
    return value

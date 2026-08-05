from __future__ import annotations

from pathlib import Path


BUILTIN_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "presets"


def resolve_config_reference(value: str | Path, base: Path) -> Path:
    """Resolve a filesystem path or a package-owned ``builtin:`` reference."""
    raw = str(value)
    if not raw.startswith("builtin:"):
        return (base / Path(raw)).resolve()

    relative = Path(raw.removeprefix("builtin:"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid builtin config reference: {raw}")
    root = BUILTIN_CONFIG_ROOT.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"builtin config reference escapes package root: {raw}")
    return path

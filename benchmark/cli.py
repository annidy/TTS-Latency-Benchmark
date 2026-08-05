from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

from . import __version__
from .config import (
    collect_required_env,
    load_config,
    missing_required_env,
)
from .models import SCHEMA_VERSION
from .adapters import registry
from .reporting import write_run_artifacts
from .runners import run_local_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttfa-bench")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="validate config, credentials, and runtime")
    doctor.add_argument("--config", type=Path, required=True)
    doctor.add_argument("--env-file", type=Path)

    run = commands.add_parser("run", help="run the benchmark on this local client node")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--env-file", type=Path)
    run.add_argument("--seed", type=int, default=20260722)
    run.add_argument("--run-id")
    run.add_argument("--output-dir", type=Path)
    run.add_argument(
        "--save-audio",
        action="store_true",
        help="save successful sample audio under the run artifact directory",
    )
    run.add_argument(
        "--boxplots",
        action="store_true",
        help="also generate and display the optional SVG box plots",
    )

    commands.add_parser("models", help="list supported public model IDs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "run":
            return asyncio.run(_run(args))
        if args.command == "models":
            print("\n".join(registry.models()))
            return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2


def _doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.env_file, allow_missing_env=True)
    env = dict(os.environ)
    if args.env_file:
        from dotenv import dotenv_values

        env = {**{k: v for k, v in dotenv_values(args.env_file).items() if v is not None}, **env}
    missing = missing_required_env(config.raw, env)
    unknown = sorted(
        provider.get("kind")
        for provider in config.providers
        if provider.get("kind") not in registry.models()
    )
    report = {
        "config": str(args.config),
        "required_env": collect_required_env(config.raw),
        "missing_env": missing,
        "unknown_provider_kinds": unknown,
        "ffmpeg": shutil.which("ffmpeg"),
        "python": sys.version.split()[0],
        "status": "ok" if not missing and not unknown else "failed",
    }
    print(json.dumps(report, indent=2))
    if not missing and not unknown:
        load_config(args.config, args.env_file)
        return 0
    return 2


async def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.env_file, allow_missing_env=True)
    run_id = args.run_id or f"{config.run_id_prefix}-{uuid.uuid4().hex[:10]}"
    started = datetime.now(timezone.utc)
    samples = await run_local_benchmark(config, run_id, args.seed, retain_audio=args.save_audio)
    stamp = started.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir or config.output_base_dir / f"{stamp}-{run_id}"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": __version__,
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "clients": [client["id"] for client in config.clients],
        "providers": [provider["id"] for provider in config.providers],
        "profiles": [profile["id"] for profile in config.profiles],
        "sample_count": len(samples),
    }
    write_run_artifacts(
        run_dir,
        config,
        samples,
        manifest,
        save_audio=args.save_audio,
        include_boxplots=args.boxplots,
    )
    print(run_dir.resolve())
    return (
        0
        if all(
            sample.status == "completed" or sample.metadata.get("retry_recovered")
            for sample in samples
        )
        else 1
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


if __name__ == "__main__":
    raise SystemExit(main())

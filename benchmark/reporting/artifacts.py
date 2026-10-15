from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
from statistics import fmean, pstdev
from typing import Any, Iterable
import wave

import yaml

from .boxplots import write_latency_boxplots
from ..config import LoadedConfig
from .dumbbells import write_latency_dumbbell
from ..models import SampleResult


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def aggregate(samples: Iterable[SampleResult]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[SampleResult]] = defaultdict(list)
    for sample in samples:
        buckets[(sample.client_id, sample.profile_id, sample.provider_id)].append(sample)
    rows = []
    for (client, profile, provider), group in sorted(buckets.items()):
        successful = [sample for sample in group if sample.status == "completed"]
        ttfa = [float(sample.ttfa_ms) for sample in successful if sample.ttfa_ms is not None]
        ttfb = [float(sample.ttfb_ms) for sample in successful if sample.ttfb_ms is not None]
        silence = [
            float(sample.initial_silence_ms)
            for sample in successful
            if sample.initial_silence_ms is not None
        ]
        rows.append(
            {
                "client_id": client,
                "client_region": group[0].client_region,
                "profile_id": profile,
                "provider_id": provider,
                "transport": group[0].transport,
                "protocol": group[0].protocol,
                "connection_mode": group[0].connection_mode,
                "connection_lifecycle": group[0]
                .metadata.get("measurement_protocol", {})
                .get("connection_lifecycle", "unspecified"),
                "connection_setup_in_timing": group[0]
                .metadata.get("measurement_protocol", {})
                .get("connection_setup_in_timing"),
                "protocol_framing": group[0]
                .metadata.get("measurement_protocol", {})
                .get("framing", "unspecified"),
                "request_start_boundary": group[0]
                .metadata.get("measurement_protocol", {})
                .get("request_start_boundary", "unspecified"),
                "first_response_boundary": group[0]
                .metadata.get("measurement_protocol", {})
                .get("first_response_boundary", "unspecified"),
                "completion_boundary": group[0]
                .metadata.get("measurement_protocol", {})
                .get("completion_boundary", "unspecified"),
                "model_id": group[0]
                .metadata.get("provider_settings", {})
                .get("model", {})
                .get("id", "unknown"),
                "model_tier": group[0]
                .metadata.get("provider_settings", {})
                .get("model", {})
                .get("tier", "unspecified"),
                "service_region": group[0]
                .metadata.get("provider_settings", {})
                .get("endpoint", {})
                .get("service_region", "undisclosed"),
                "attempts": len(group),
                "successes": len(successful),
                "success_rate": len(successful) / len(group),
                "ttfa_mean_ms": fmean(ttfa) if ttfa else None,
                "ttfa_stddev_ms": pstdev(ttfa) if len(ttfa) > 1 else 0.0 if ttfa else None,
                "ttfa_p50_ms": percentile(ttfa, 0.50),
                "ttfa_p75_ms": percentile(ttfa, 0.75),
                "ttfa_p90_ms": percentile(ttfa, 0.90),
                "ttfa_p95_ms": percentile(ttfa, 0.95),
                "ttfa_p99_ms": percentile(ttfa, 0.99),
                "ttfb_p50_ms": percentile(ttfb, 0.50),
                "ttfb_p95_ms": percentile(ttfb, 0.95),
                "initial_silence_p50_ms": percentile(silence, 0.50),
                "initial_silence_p95_ms": percentile(silence, 0.95),
            }
        )
    return rows


def write_run_artifacts(
    run_dir: Path,
    config: LoadedConfig,
    samples: list[SampleResult],
    manifest: dict[str, Any],
    *,
    save_audio: bool = False,
    include_boxplots: bool = False,
) -> list[dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = aggregate(samples)
    if save_audio:
        _write_audio_artifacts(run_dir, samples)
    (run_dir / "config.redacted.yaml").write_text(
        yaml.safe_dump(config.redacted(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    report_samples = _preferred_report_samples(samples)
    dumbbell_path = run_dir / "dumbbells" / "latency.svg"
    write_latency_dumbbell(
        dumbbell_path,
        report_samples,
        group_label=lambda sample: sample.provider_id.rsplit("/", 1)[0],
        title=str(manifest["run_id"]),
    )
    boxplots = (
        write_latency_boxplots(
            run_dir / "boxplots",
            report_samples,
            group_label=lambda sample: sample.provider_id.rsplit("/", 1)[0],
            title_prefix=str(manifest["run_id"]),
        )
        if include_boxplots
        else {}
    )
    (run_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(run_dir / "summary.csv", rows)
    plot_links = {metric: path.relative_to(run_dir).as_posix() for metric, path in boxplots.items()}
    (run_dir / "report.md").write_text(
        _markdown_report(
            manifest,
            rows,
            dumbbell_path.relative_to(run_dir).as_posix(),
            plot_links,
        ),
        encoding="utf-8",
    )
    (run_dir / "report_zh.md").write_text(
        _markdown_report_zh(
            manifest,
            rows,
            dumbbell_path.relative_to(run_dir).as_posix(),
            plot_links,
        ),
        encoding="utf-8",
    )
    return rows


def _write_audio_artifacts(run_dir: Path, samples: list[SampleResult]) -> None:
    audio_dir = run_dir / "audio"
    for sample in samples:
        payload = sample.audio_payload
        fmt = sample.audio_format or {}
        if sample.status != "completed" or not payload:
            continue
        repetition = sample.metadata.get("repetition", "unknown")
        stem = _safe_name(
            f"{sample.client_region}__{sample.provider_id}__{sample.text_id}__r{repetition}"
        )
        encoding = str(fmt.get("encoding", "unknown")).lower()
        if encoding in {"pcm", "pcm_s16le", "pcm16", "raw"}:
            path = audio_dir / f"{stem}.wav"
            audio_dir.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as output:
                output.setnchannels(int(fmt.get("channels", 1)))
                output.setsampwidth(int(fmt.get("sample_width_bytes", 2)))
                output.setframerate(int(fmt["sample_rate_hz"]))
                output.writeframes(payload)
        else:
            suffix = ".wav" if encoding in {"wav", "wave"} else f".{_safe_name(encoding)}"
            path = audio_dir / f"{stem}{suffix}"
            audio_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        sample.metadata["audio_artifact"] = path.relative_to(run_dir).as_posix()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    dumbbell: str | None = None,
    boxplots: dict[str, str] | None = None,
) -> str:
    lines = [
        f"# TTFA Benchmark: {manifest['run_id']}",
        "",
        f"- Started: `{manifest['started_at']}`",
        f"- Execution-order seed: `{manifest['seed']}` (does not control model generation)",
        f"- Schema: `{manifest['schema_version']}`",
        "",
        "| Model | Protocol | Region | Success rate | TTFB p50 | TTFB p95 | Silence p50 | Silence p95 | TTFA p50 | TTFA p95 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display_rows = _preferred_report_rows(rows)
    for row in sorted(
        display_rows,
        key=lambda item: (item["client_id"], item["profile_id"], item["ttfa_p50_ms"] or math.inf),
    ):
        lines.append(
            "| {model} | {protocol} | {client_region} | {success} | {ttfb_p50} | {ttfb_p95} | {silence_p50} | {silence_p95} | {ttfa_p50} | {ttfa_p95} |".format(
                **row,
                model=row["provider_id"].rsplit("/", 1)[0],
                success=f"{row['successes']}/{row['attempts']} ({row['success_rate']:.0%})",
                ttfb_p50=_ms(row["ttfb_p50_ms"]),
                ttfb_p95=_ms(row["ttfb_p95_ms"]),
                silence_p50=_ms(row["initial_silence_p50_ms"]),
                silence_p95=_ms(row["initial_silence_p95_ms"]),
                ttfa_p50=_ms(row["ttfa_p50_ms"]),
                ttfa_p95=_ms(row["ttfa_p95_ms"]),
            )
        )
    lines.extend(
        [
            "",
            "The display table keeps persistent WebSocket when a model has both WebSocket and HTTP results; HTTP is shown only when WebSocket is unavailable.",
            "TTFB is the arrival of the first non-empty audio chunk; metadata and control messages never qualify.",
            "Full metrics and protocol metadata remain available in `summary.json`, `summary.csv`, and `samples.jsonl`.",
            "",
        ]
    )
    if dumbbell:
        lines.extend(
            [
                "## Latency comparison",
                "",
                "Models are sorted from top to bottom by TTFA p50; lower is faster.",
                "",
                f"![TTFB p50, TTFA p50, and TTFA p95]({dumbbell})",
                "",
            ]
        )
    if boxplots:
        lines.extend(["## Latency distributions", ""])
        for metric, label in (("ttfb", "TTFB"), ("silence", "Silence"), ("ttfa", "TTFA")):
            if metric in boxplots:
                lines.extend([f"### {label}", "", f"![{label} box plot]({boxplots[metric]})", ""])
    return "\n".join(lines)

def _markdown_report_zh(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    dumbbell: str | None = None,
    boxplots: dict[str, str] | None = None,
) -> str:
    lines = [
        f"# TTFA 基准测试：{manifest['run_id']}",
        "",
        f"- 开始时间：`{manifest['started_at']}`",
        f"- 执行顺序种子：`{manifest['seed']}`（不控制模型生成）",
        f"- Schema：`{manifest['schema_version']}`",
        "",
        "| 模型 | 协议 | 区域 | 成功率 | TTFB p50 | TTFB p95 | 静音 p50 | 静音 p95 | TTFA p50 | TTFA p95 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display_rows = _preferred_report_rows(rows)
    for row in sorted(
        display_rows,
        key=lambda item: (item["client_id"], item["profile_id"], item["ttfa_p50_ms"] or math.inf),
    ):
        lines.append(
            "| {model} | {protocol} | {client_region} | {success} | {ttfb_p50} | {ttfb_p95} | {silence_p50} | {silence_p95} | {ttfa_p50} | {ttfa_p95} |".format(
                **row,
                model=row["provider_id"].rsplit("/", 1)[0],
                success=f"{row['successes']}/{row['attempts']} ({row['success_rate']:.0%})",
                ttfb_p50=_ms(row["ttfb_p50_ms"]),
                ttfb_p95=_ms(row["ttfb_p95_ms"]),
                silence_p50=_ms(row["initial_silence_p50_ms"]),
                silence_p95=_ms(row["initial_silence_p95_ms"]),
                ttfa_p50=_ms(row["ttfa_p50_ms"]),
                ttfa_p95=_ms(row["ttfa_p95_ms"]),
            )
        )
    lines.extend(
        [
            "",
            "展示表在同一模型同时有 WebSocket 与 HTTP 结果时优先保留持久 WebSocket；仅当 WebSocket 不可用时才展示 HTTP。",
            "TTFB 为首个非空音频块到达时间，元数据与控制消息不计入；TTFA 为首个音频包到达时间。",
            "完整指标与协议元数据见 `summary.json`、`summary.csv` 和 `samples.jsonl`。",
            "",
        ]
    )
    if dumbbell:
        lines.extend(
            [
                "## 延迟对比",
                "",
                "模型按 TTFA p50 从上到下排序；数值越低越快。",
                "",
                f"![TTFB p50、TTFA p50 与 TTFA p95]({dumbbell})",
                "",
            ]
        )
    if boxplots:
        lines.extend(["## 延迟分布", ""])
        for metric, label in (("ttfb", "TTFB"), ("silence", "静音"), ("ttfa", "TTFA")):
            if metric in boxplots:
                lines.extend([f"### {label}", "", f"![{label} 箱线图]({boxplots[metric]})", ""])
    return "\n".join(lines)

def _preferred_report_samples(samples: list[SampleResult]) -> list[SampleResult]:
    def key(sample: SampleResult) -> tuple[str, str, str]:
        return (
            sample.client_id,
            sample.client_region,
            sample.provider_id.rsplit("/", 1)[0],
        )

    websocket_keys = {key(sample) for sample in samples if sample.transport == "websocket"}
    return [
        sample
        for sample in samples
        if key(sample) not in websocket_keys or sample.transport == "websocket"
    ]


def _preferred_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one display protocol per client/model while retaining all summary rows.

    A model is identified by the canonical portion of ``provider_id`` before its
    transport suffix. If any WebSocket row exists for that client, region and model,
    every HTTP row for the same key is hidden from Markdown. Multiple rows using the
    selected protocol (for example distinct execution profiles) remain visible.
    """

    def key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(row["client_id"]),
            str(row["client_region"]),
            str(row["provider_id"]).rsplit("/", 1)[0],
        )

    websocket_keys = {key(row) for row in rows if row.get("transport") == "websocket"}
    return [
        row for row in rows if key(row) not in websocket_keys or row.get("transport") == "websocket"
    ]


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} ms"

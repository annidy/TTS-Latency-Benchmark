from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from .artifacts import percentile
from .boxplots import write_latency_boxplots
from .dumbbells import write_latency_dumbbell


class Region(Protocol):
    """Minimal region metadata required by distributed report rendering."""

    key: str
    label: str
    locations: tuple[str, ...]
    datacenters: tuple[str, ...]


MODEL_NAMES = {
    "elevenlabs.flash-v2.5": "ElevenLabs Flash v2.5",
    "fish-audio.s2.1-pro": "Fish Audio S2.1 Pro",
    "breezeblue.breeze-tts-2": "BreezeBlue Breeze TTS 2",
    "cartesia.sonic-3.5": "Cartesia Sonic 3.5",
    "inworld.tts-2": "Inworld TTS-2",
    "xai.tts": "xAI TTS",
    "elevenlabs.v3": "ElevenLabs v3",
    "async.flash-v1.5": "Async Flash 1.5",
    "speechify.simba-3.2": "Speechify Simba 3.2",
}

METRICS = (
    ("ttfb", "TTFB"),
    ("initial_silence", "Silence"),
    ("ttfa", "TTFA"),
)
PERCENTILES = ("p50", "p95")
ONSET_ALGORITHMS = ("v1", "v2")


def canonical_id(row: dict[str, Any]) -> str:
    return str(row["provider_id"]).rsplit("/", 1)[0]


def _read_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"missing replica samples: {path}")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _read_replica_samples(
    root: Path,
    region_key: str,
    replica: int,
) -> list[dict[str, Any]]:
    relative = Path(region_key) / f"cpu-{replica}" / "samples.jsonl"
    paths = [root / relative]
    supplements = root / "supplements"
    if supplements.exists():
        paths.extend(sorted(supplements.glob(f"*/{relative.as_posix()}")))
    samples = [sample for path in paths for sample in _read_samples(path)]
    overrides = root / "overrides"
    override_paths = (
        sorted(overrides.glob(f"*/{relative.as_posix()}")) if overrides.exists() else []
    )
    override_samples = [sample for path in override_paths for sample in _read_samples(path)]
    override_models = {canonical_id(sample) for sample in override_samples}
    return [
        sample for sample in samples if canonical_id(sample) not in override_models
    ] + override_samples


def _successful_samples_by_region(
    root: Path,
    regions: tuple[Region, ...],
    replicas_per_region: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for region in regions:
        result[region.key] = [
            sample
            for replica in range(1, replicas_per_region + 1)
            for sample in _read_replica_samples(root, region.key, replica)
            if sample.get("status") == "completed"
        ]
    return result


def _aggregate_model(samples: list[dict[str, Any]]) -> dict[str, Any]:
    first = samples[0]
    row: dict[str, Any] = {
        "provider_id": first["provider_id"],
        "transport": first["transport"],
        "protocol": first["protocol"],
        "attempts": len(samples),
        "successes": len(samples),
        "success_rate": 1.0,
        "client_regions": sorted({str(sample["client_region"]) for sample in samples}),
        "_metric_values": {},
    }
    for metric, _ in METRICS:
        source = "initial_silence_ms" if metric == "initial_silence" else f"{metric}_ms"
        values = [float(sample[source]) for sample in samples]
        row["_metric_values"][metric] = values
        row[f"{metric}_p50_ms"] = percentile(values, 0.50)
        row[f"{metric}_p95_ms"] = percentile(values, 0.95)
    for algorithm in ONSET_ALGORITHMS:
        for metric in ("initial_silence", "ttfa"):
            variant_source = f"{metric}_{algorithm}_ms"
            fallback_source = f"{metric}_ms"
            values = []
            for sample in samples:
                value = (
                    sample.get(variant_source)
                    if variant_source in sample
                    else sample[fallback_source]
                )
                if value is not None:
                    values.append(float(value))
            if not values:
                raise ValueError(
                    f"no usable {algorithm} values for {first['provider_id']}/{metric}"
                )
            row["_metric_values"][f"{metric}_{algorithm}"] = values
            row[f"{metric}_{algorithm}_count"] = len(values)
            row[f"{metric}_{algorithm}_p50_ms"] = percentile(values, 0.50)
            row[f"{metric}_{algorithm}_p95_ms"] = percentile(values, 0.95)
    return row


def load_results(
    root: Path,
    regions: tuple[Region, ...],
    expected_attempts_per_replica: int = 20,
    replicas_per_region: int = 3,
) -> dict[str, dict[str, dict[str, Any]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    sample_overrides_path = root / "model-sample-overrides.json"
    sample_overrides = (
        json.loads(sample_overrides_path.read_text(encoding="utf-8"))
        if sample_overrides_path.exists()
        else {}
    )
    if not isinstance(sample_overrides, dict):
        raise ValueError("model-sample-overrides.json must contain an object")
    expected_models = {
        canonical_id(sample) for sample in _read_replica_samples(root, regions[0].key, 1)
    }
    unknown = expected_models - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"missing report display names for models: {sorted(unknown)}")
    for region in regions:
        pooled: dict[str, list[dict[str, Any]]] = {model: [] for model in expected_models}
        for replica in range(1, replicas_per_region + 1):
            replica_samples = _read_replica_samples(root, region.key, replica)
            by_model: dict[str, list[dict[str, Any]]] = {}
            for sample in replica_samples:
                by_model.setdefault(canonical_id(sample), []).append(sample)
            if set(by_model) != expected_models:
                missing = sorted(expected_models - set(by_model))
                extra = sorted(set(by_model) - expected_models)
                raise ValueError(
                    f"{region.key}/cpu-{replica}: model mismatch; missing={missing}, extra={extra}"
                )
            for model, samples in by_model.items():
                expected_model_attempts = int(
                    sample_overrides.get(model, expected_attempts_per_replica)
                )
                successful = [sample for sample in samples if sample.get("status") == "completed"]
                unrecovered = [
                    sample
                    for sample in samples
                    if sample.get("status") != "completed"
                    and not sample.get("metadata", {}).get("retry_recovered")
                ]
                if len(successful) != expected_model_attempts:
                    raise ValueError(
                        f"{region.key}/cpu-{replica}/{model}: expected "
                        f"{expected_model_attempts} successful measurements, "
                        f"got {len(successful)} from {len(samples)} attempts"
                    )
                if unrecovered:
                    raise ValueError(
                        f"{region.key}/cpu-{replica}/{model}: "
                        f"{len(unrecovered)} measurement failures were not recovered"
                    )
                client_regions = {
                    str(sample.get("client_region", "")).upper() for sample in successful
                }
                if not client_regions.issubset({location.upper() for location in region.locations}):
                    raise ValueError(
                        f"{region.key}/cpu-{replica}/{model}: unexpected "
                        f"client regions {sorted(client_regions)}"
                    )
                pooled[model].extend(successful)
        by_model = {model: _aggregate_model(samples) for model, samples in pooled.items()}
        for model, row in by_model.items():
            expected_region_attempts = (
                int(sample_overrides.get(model, expected_attempts_per_replica))
                * replicas_per_region
            )
            if int(row["attempts"]) != expected_region_attempts:
                raise ValueError(
                    f"{region.key}/{model}: expected {expected_region_attempts} pooled "
                    f"attempts, got {row['attempts']}"
                )
        matrix[region.key] = by_model
    return matrix


def protocol_label(row: dict[str, Any]) -> str:
    return "WS persistent" if row.get("transport") == "websocket" else "HTTP warm pool"


def flat_rows(
    matrix: dict[str, dict[str, dict[str, Any]]], regions: tuple[Region, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, display_name in MODEL_NAMES.items():
        if model not in matrix[regions[0].key]:
            continue
        output: dict[str, Any] = {
            "model": display_name,
            "model_id": model,
            "protocol": protocol_label(matrix[regions[0].key][model]),
        }
        for region in regions:
            regional = matrix[region.key][model]
            output[f"{region.key}_attempts"] = regional["attempts"]
            output[f"{region.key}_successes"] = regional["successes"]
            for metric, _ in METRICS:
                for quantile in PERCENTILES:
                    field = f"{metric}_{quantile}_ms"
                    output[f"{region.key}_{field}"] = regional[field]
            for algorithm in ONSET_ALGORITHMS:
                for metric in ("initial_silence", "ttfa"):
                    count_field = f"{metric}_{algorithm}_count"
                    output[f"{region.key}_{count_field}"] = regional[count_field]
                    for quantile in PERCENTILES:
                        field = f"{metric}_{algorithm}_{quantile}_ms"
                        output[f"{region.key}_{field}"] = regional[field]
        for metric, _ in METRICS:
            for quantile in PERCENTILES:
                field = f"{metric}_{quantile}_ms"
                output[f"avg_{field}"] = mean(
                    float(matrix[region.key][model][field]) for region in regions
                )
                combined_values = [
                    float(value)
                    for region in regions
                    for value in matrix[region.key][model]["_metric_values"][metric]
                ]
                output[f"combined_{field}"] = percentile(
                    combined_values, 0.50 if quantile == "p50" else 0.95
                )
        for algorithm in ONSET_ALGORITHMS:
            for metric in ("initial_silence", "ttfa"):
                for quantile in PERCENTILES:
                    field = f"{metric}_{algorithm}_{quantile}_ms"
                    output[f"avg_{field}"] = mean(
                        float(matrix[region.key][model][field]) for region in regions
                    )
                    combined_values = [
                        float(value)
                        for region in regions
                        for value in matrix[region.key][model]["_metric_values"][
                            f"{metric}_{algorithm}"
                        ]
                    ]
                    output[f"combined_{field}"] = percentile(
                        combined_values, 0.50 if quantile == "p50" else 0.95
                    )
        rows.append(output)
    return sorted(rows, key=lambda row: float(row["combined_ttfa_p50_ms"]))


def _latency_table(
    rows: list[dict[str, Any]],
    algorithm: str,
    quantile: str,
    regions: tuple[Region, ...],
) -> str:
    header_regions = "".join(
        f'<th colspan="3">{region.label} {algorithm} {quantile}</th>' for region in regions
    )
    header_metrics = "".join(
        f"<th>{label} {quantile}</th>" for _ in range(len(regions) + 1) for _, label in METRICS
    )
    body: list[str] = []
    for row in rows:
        values: list[str] = []
        for region in regions:
            for metric, _ in METRICS:
                field = (
                    f"{region.key}_{metric}_{quantile}_ms"
                    if metric == "ttfb"
                    else f"{region.key}_{metric}_{algorithm}_{quantile}_ms"
                )
                values.append(f'<td align="right">{float(row[field]):.1f} ms</td>')
        for metric, _ in METRICS:
            field = (
                f"combined_{metric}_{quantile}_ms"
                if metric == "ttfb"
                else f"combined_{metric}_{algorithm}_{quantile}_ms"
            )
            values.append(f'<td align="right"><strong>{float(row[field]):.1f} ms</strong></td>')
        body.append(
            f"<tr><td>{row['model']}</td><td>{row['protocol']}</td>" + "".join(values) + "</tr>"
        )
    body_rows = "\n    ".join(body)
    return f"""<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Protocol</th>
      {header_regions}
      <th colspan="3">Combined {algorithm} {quantile}</th>
    </tr>
    <tr>{header_metrics}</tr>
  </thead>
  <tbody>
    {body_rows}
  </tbody>
</table>"""


def render_report(
    matrix: dict[str, dict[str, dict[str, Any]]],
    regions: tuple[Region, ...],
    title: str,
    expected_attempts_per_replica: int = 20,
    replicas_per_region: int = 3,
    region_execution: str = "parallel",
    cpu_execution: str = "parallel",
    model_execution: str = "serial",
    request_execution: str = "serial",
    orchestration: list[dict[str, Any]] | None = None,
    methodology_notes: list[str] | None = None,
    model_order: str = "configured",
    dumbbell_links: dict[str, str] | None = None,
    boxplot_links: dict[str, dict[str, str]] | None = None,
) -> str:
    rows = flat_rows(matrix, regions)
    expected_region_attempts = expected_attempts_per_replica * replicas_per_region
    compute_lines = "\n".join(
        f"- {item['region']} CPU {item['replica']} "
        f"(`{item.get('location', item.get('datacenter', 'unknown'))}`): {item['compute']}"
        for item in (orchestration or [])
    )
    compute_section = f"\n## Client compute\n\n{compute_lines}\n" if compute_lines else ""
    notes_lines = "\n".join(f"- {note}" for note in (methodology_notes or []))
    notes_section = f"\n## Model-specific execution notes\n\n{notes_lines}\n" if notes_lines else ""
    dumbbells_section = _dumbbell_report_section(dumbbell_links or {}, regions)
    boxplots_section = _boxplot_report_section(boxplot_links or {}, regions)
    region_labels = ", ".join(region.label for region in regions)
    return f"""# {title}

- Regions: {region_labels}. Region execution is {region_execution}.
- Each region has {replicas_per_region} independent CPU clients; CPU execution is {cpu_execution}.
- Models on each CPU run {model_execution} in {model_order} order; measured requests for each model run {request_execution}.
- Every model receives three excluded warmup requests immediately before its measured batch. Persistent WebSocket models reuse that warmed connection.
- Provider quota errors during warmup wait and retry. Failed measured attempts are retained for audit, wait before retrying, and never enter the pooled latency distribution.
- Each CPU contributes {expected_attempts_per_replica} measured requests per model. Regional p50/p95 are calculated directly from all {expected_region_attempts} pooled raw samples, not from averaged per-CPU percentiles.
- Unless a model-specific note states otherwise, every reported regional cell contains exactly {expected_region_attempts} successful measured samples.
- One selected protocol is reported per model using the public labels `WebSocket` or `HTTP`.
- TTFB means the first non-empty audio chunk; metadata/control/context frames do not count.
- Every request is evaluated by both onset detectors. v1 uses the fixed absolute-density detector; v2 uses the adaptive energy-hysteresis detector. TTFB is identical in both table pairs, while Silence and TTFA use the named detector.
- Combined p50/p95 is recalculated from the pooled successful raw samples across all regions, matching the All regions combined charts.
{compute_section}
{notes_section}
{dumbbells_section}
{boxplots_section}

## v1 p50 results

{_latency_table(rows, "v1", "p50", regions)}

## v1 p95 results

{_latency_table(rows, "v1", "p95", regions)}

## v2 p50 results

{_latency_table(rows, "v2", "p50", regions)}

## v2 p95 results

{_latency_table(rows, "v2", "p95", regions)}
"""


def _dumbbell_report_section(links: dict[str, str], regions: tuple[Region, ...]) -> str:
    if not links:
        return ""
    labels = {region.key: region.label for region in regions}
    labels["combined"] = "All regions combined"
    lines = [
        "## Latency comparison",
        "",
        "Each chart shows TTFB p50, TTFA p50, and TTFA p95. Models are sorted "
        "from top to bottom by TTFA p50; lower is faster.",
        "",
    ]
    for scope in [*(region.key for region in regions), "combined"]:
        if scope not in links:
            continue
        lines.extend(
            [
                f"### {labels[scope]}",
                "",
                f"![{labels[scope]} latency comparison]({links[scope]})",
                "",
            ]
        )
    return "\n".join(lines)


def _boxplot_report_section(links: dict[str, dict[str, str]], regions: tuple[Region, ...]) -> str:
    if not links:
        return ""
    labels = {region.key: region.label for region in regions}
    labels["combined"] = "All regions combined"
    lines = [
        "## Latency distributions",
        "",
        "Each chart is independently ordered from left to right by the plotted metric's p50. ",
        "Boxes span p25–p75, the center line is p50, whiskers use 1.5×IQR, and dots are outliers.",
        "",
    ]
    for scope in [*(region.key for region in regions), "combined"]:
        if scope not in links:
            continue
        lines.extend([f"### {labels[scope]}", ""])
        for metric, label in (("ttfb", "TTFB"), ("silence", "Silence"), ("ttfa", "TTFA")):
            if metric in links[scope]:
                lines.extend(
                    [f"#### {label}", "", f"![{labels[scope]} {label}]({links[scope][metric]})", ""]
                )
    return "\n".join(lines)


def write_combined_report(
    root: Path,
    regions: tuple[Region, ...],
    title: str,
    expected_attempts_per_replica: int = 20,
    replicas_per_region: int = 3,
    region_execution: str = "parallel",
    cpu_execution: str = "parallel",
    model_execution: str = "serial",
    request_execution: str = "serial",
    model_order: str = "configured",
    include_boxplots: bool = False,
) -> Path:
    matrix = load_results(root, regions, expected_attempts_per_replica, replicas_per_region)
    rows = flat_rows(matrix, regions)
    samples_by_region = _successful_samples_by_region(root, regions, replicas_per_region)
    dumbbell_links: dict[str, str] = {}
    for region in regions:
        path = root / "dumbbells" / f"{region.key}-latency.svg"
        write_latency_dumbbell(
            path,
            samples_by_region[region.key],
            group_label=lambda sample: MODEL_NAMES[canonical_id(sample)],
            title=f"{region.label} latency comparison",
        )
        dumbbell_links[region.key] = path.relative_to(root).as_posix()
    combined_samples = [sample for region in regions for sample in samples_by_region[region.key]]
    combined_path = root / "dumbbells" / "combined-latency.svg"
    write_latency_dumbbell(
        combined_path,
        combined_samples,
        group_label=lambda sample: MODEL_NAMES[canonical_id(sample)],
        title="All regions combined latency comparison",
    )
    dumbbell_links["combined"] = combined_path.relative_to(root).as_posix()
    boxplot_links: dict[str, dict[str, str]] = {}
    if include_boxplots:
        for region in regions:
            paths = write_latency_boxplots(
                root / "boxplots",
                samples_by_region[region.key],
                group_label=lambda sample: MODEL_NAMES[canonical_id(sample)],
                title_prefix=region.label,
                filename_prefix=f"{region.key}-",
            )
            boxplot_links[region.key] = {
                metric: path.relative_to(root).as_posix() for metric, path in paths.items()
            }
        combined_paths = write_latency_boxplots(
            root / "boxplots",
            combined_samples,
            group_label=lambda sample: MODEL_NAMES[canonical_id(sample)],
            title_prefix="All regions combined",
            filename_prefix="combined-",
        )
        boxplot_links["combined"] = {
            metric: path.relative_to(root).as_posix() for metric, path in combined_paths.items()
        }
    orchestration_path = root / "orchestration.json"
    orchestration = (
        json.loads(orchestration_path.read_text(encoding="utf-8"))
        if orchestration_path.exists()
        else None
    )
    methodology_notes_path = root / "methodology-notes.json"
    methodology_notes = (
        json.loads(methodology_notes_path.read_text(encoding="utf-8"))
        if methodology_notes_path.exists()
        else None
    )
    report_path = root / "report.md"
    report_path.write_text(
        render_report(
            matrix,
            regions,
            title,
            expected_attempts_per_replica,
            replicas_per_region,
            region_execution,
            cpu_execution,
            model_execution,
            request_execution,
            orchestration=orchestration,
            methodology_notes=methodology_notes,
            model_order=model_order,
            dumbbell_links=dumbbell_links,
            boxplot_links=boxplot_links,
        ),
        encoding="utf-8",
    )
    (root / "combined-summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (root / "combined-summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report_path

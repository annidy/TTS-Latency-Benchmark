from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from html import escape
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class LatencyDumbbell:
    ttfb_p50_ms: float
    ttfa_p50_ms: float
    ttfa_p95_ms: float


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        raise ValueError("percentile requires at least one finite value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def latency_dumbbell_stats(
    samples: Sequence[Any], *, group_label: Callable[[Any], str]
) -> dict[str, LatencyDumbbell]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"ttfb": [], "ttfa": []})
    for sample in samples:
        status = sample.get("status") if isinstance(sample, dict) else sample.status
        if status != "completed":
            continue
        ttfb = sample.get("ttfb_ms") if isinstance(sample, dict) else sample.ttfb_ms
        ttfa = sample.get("ttfa_ms") if isinstance(sample, dict) else sample.ttfa_ms
        if ttfb is None or ttfa is None:
            continue
        label = group_label(sample)
        grouped[label]["ttfb"].append(float(ttfb))
        grouped[label]["ttfa"].append(float(ttfa))
    return {
        label: LatencyDumbbell(
            ttfb_p50_ms=_percentile(values["ttfb"], 0.50),
            ttfa_p50_ms=_percentile(values["ttfa"], 0.50),
            ttfa_p95_ms=_percentile(values["ttfa"], 0.95),
        )
        for label, values in grouped.items()
        if values["ttfb"] and values["ttfa"]
    }


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    nice = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    return nice * (10**exponent)


def write_latency_dumbbell_svg(
    path: Path,
    series: Mapping[str, LatencyDumbbell],
    *,
    title: str,
) -> list[str]:
    """Write a horizontal latency dumbbell chart sorted by TTFA p50."""

    rows = sorted(series.items(), key=lambda item: (item[1].ttfa_p50_ms, item[0]))
    if not rows:
        raise ValueError("cannot render a latency dumbbell chart without samples")

    row_height = 76
    top = 150
    bottom = 42
    height = top + row_height * len(rows) + bottom
    label_x = 20
    label_gap = 30
    estimated_label_width = max(len(label) * 11 for label, _ in rows)
    plot_left = max(225, label_x + estimated_label_width + label_gap)
    plot_width = 895
    plot_right = plot_left + plot_width
    values_x = plot_right + 40
    width = values_x + 320
    x_max = _nice_ceiling(max(item.ttfa_p95_ms for _, item in rows) * 1.05)
    tick_count = 4

    def x(value: float) -> float:
        return plot_left + (plot_right - plot_left) * max(0.0, min(value, x_max)) / x_max

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        '<desc id="desc">Models sorted from top to bottom by TTFA p50, lower is faster. '
        "Each row shows TTFB p50, TTFA p50, and TTFA p95 in milliseconds.</desc>",
        "<style>.background{fill:#fff}text{font-family:Arial,sans-serif;fill:#000}"
        ".title{font-size:24px;font-weight:600}"
        ".note,.legend,.tick,.column{fill:#000}.note{font-size:15px}.legend{font-size:15px}"
        ".tick{font-size:13px}.column{font-size:14px}.model{font-size:18px;font-weight:600}"
        ".value{font-size:17px}.grid,.row{stroke:#d9d9d9;stroke-width:1}"
        ".lead{stroke:#777;stroke-width:3;stroke-linecap:round}"
        ".range{stroke:#6eb6f2;stroke-width:3;stroke-linecap:round}.ttfb{fill:#777}"
        ".ttfa50{fill:#338bd1;stroke:#338bd1;stroke-width:2}"
        ".ttfa95{fill:none;stroke:#338bd1;stroke-width:3}</style>",
        f'<rect class="background" x="0" y="0" width="{width}" height="{height}"/>',
        f'<text class="title" x="{label_x}" y="34">TTFB p50 · TTFA p50 · TTFA p95</text>',
        f'<text class="note" x="{width - 20}" y="34" text-anchor="end">'
        "Sorted by TTFA p50 · lower is faster</text>",
        f'<rect class="ttfb" x="{plot_left}" y="69" width="10" height="10" '
        f'transform="rotate(45 {plot_left + 5} 74)"/>',
        f'<text class="legend" x="{plot_left + 24}" y="80">TTFB p50</text>',
        f'<circle class="ttfa50" cx="{plot_left + 155}" cy="74" r="7"/>',
        f'<text class="legend" x="{plot_left + 174}" y="80">TTFA p50</text>',
        f'<circle class="ttfa95" cx="{plot_left + 300}" cy="74" r="7"/>',
        f'<text class="legend" x="{plot_left + 316}" y="80">TTFA p95</text>',
        f'<text class="column" x="{values_x}" y="116">TTFB / TTFA p50 / p95</text>',
    ]
    for index in range(tick_count + 1):
        value = x_max * index / tick_count
        xx = x(value)
        parts.extend(
            [
                f'<line class="grid" x1="{xx:.2f}" y1="{top - 24}" '
                f'x2="{xx:.2f}" y2="{height - bottom}"/>',
                f'<text class="tick" x="{xx:.2f}" y="112" text-anchor="middle">{value:.0f}</text>',
            ]
        )

    for index, (label, item) in enumerate(rows):
        cy = top + row_height * index + row_height / 2
        ttfb_x = x(item.ttfb_p50_ms)
        ttfa50_x = x(item.ttfa_p50_ms)
        ttfa95_x = x(item.ttfa_p95_ms)
        lead_line = (
            f'<line class="lead" x1="{ttfb_x + 7:.2f}" y1="{cy:.2f}" '
            f'x2="{ttfa50_x - 7:.2f}" y2="{cy:.2f}"/>'
            if ttfa50_x - ttfb_x > 14
            else ""
        )
        range_line = (
            f'<line class="range" x1="{ttfa50_x + 7:.2f}" y1="{cy:.2f}" '
            f'x2="{ttfa95_x - 7:.2f}" y2="{cy:.2f}"/>'
            if ttfa95_x - ttfa50_x > 14
            else ""
        )
        parts.extend(
            [
                f'<g class="dumbbell" data-label="{escape(label)}" '
                f'data-ttfb-p50-ms="{item.ttfb_p50_ms:.6f}" '
                f'data-ttfa-p50-ms="{item.ttfa_p50_ms:.6f}" '
                f'data-ttfa-p95-ms="{item.ttfa_p95_ms:.6f}">',
                f'<text class="model" x="{label_x}" y="{cy + 6:.2f}">{escape(label)}</text>',
                f'<line class="row" x1="{plot_left}" y1="{cy + row_height / 2 - 1:.2f}" '
                f'x2="{plot_right}" y2="{cy + row_height / 2 - 1:.2f}"/>',
                lead_line,
                range_line,
                f'<rect class="ttfb" x="{ttfb_x - 5:.2f}" y="{cy - 5:.2f}" width="10" '
                f'height="10" transform="rotate(45 {ttfb_x:.2f} {cy:.2f})"/>',
                f'<circle class="ttfa50" cx="{ttfa50_x:.2f}" cy="{cy:.2f}" r="7"/>',
                f'<circle class="ttfa95" cx="{ttfa95_x:.2f}" cy="{cy:.2f}" r="7"/>',
                f'<text class="value" x="{values_x}" y="{cy + 6:.2f}">'
                f"{item.ttfb_p50_ms:.1f} / {item.ttfa_p50_ms:.1f} / "
                f"{item.ttfa_p95_ms:.1f} ms</text>",
                "</g>",
            ]
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return [label for label, _ in rows]


def write_latency_dumbbell(
    path: Path,
    samples: Sequence[Any],
    *,
    group_label: Callable[[Any], str],
    title: str,
) -> list[str]:
    return write_latency_dumbbell_svg(
        path,
        latency_dumbbell_stats(samples, group_label=group_label),
        title=title,
    )

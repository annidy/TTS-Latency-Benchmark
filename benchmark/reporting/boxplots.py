from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from html import escape
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class BoxStats:
    q1: float
    median: float
    q3: float
    whisker_low: float
    whisker_high: float
    outliers: tuple[float, ...]
    count: int


LATENCY_METRICS = {
    "ttfb": ("ttfb_ms", "TTFB"),
    "silence": ("initial_silence_ms", "Silence"),
    "ttfa": ("ttfa_ms", "TTFA"),
}
LATENCY_Y_MAX_MS = {"ttfb": 1000.0, "silence": 500.0, "ttfa": 1500.0}


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def box_stats(values: Sequence[float]) -> BoxStats:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        raise ValueError("box plot series must contain at least one finite value")
    q1 = _percentile(ordered, 0.25)
    median = _percentile(ordered, 0.50)
    q3 = _percentile(ordered, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inliers = [value for value in ordered if lower_fence <= value <= upper_fence]
    return BoxStats(
        q1=q1,
        median=median,
        q3=q3,
        whisker_low=inliers[0],
        whisker_high=inliers[-1],
        outliers=tuple(value for value in ordered if value < lower_fence or value > upper_fence),
        count=len(ordered),
    )


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    nice = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    return nice * (10**exponent)


def write_boxplot_svg(
    path: Path,
    series: Mapping[str, Sequence[float]],
    *,
    title: str,
    y_label: str,
    y_max_ms: float | None = None,
) -> list[str]:
    """Write a dependency-free SVG box plot and return its p50-sorted labels."""

    stats = [(label, box_stats(values)) for label, values in series.items() if values]
    stats.sort(key=lambda item: (item[1].median, item[0]))
    if not stats:
        raise ValueError("cannot render a box plot without samples")

    width = max(960, 150 + 115 * len(stats))
    height = 650
    left, right, top, bottom = 105, 40, 70, 185
    plot_width = width - left - right
    plot_height = height - top - bottom
    observed_max = max(
        max(item.whisker_high, item.q3, *(item.outliers or (item.q3,))) for _, item in stats
    )
    y_max = float(y_max_ms) if y_max_ms is not None else _nice_ceiling(observed_max * 1.05)
    if y_max <= 0:
        raise ValueError("box plot y-axis maximum must be positive")

    def y(value: float) -> float:
        clipped = min(y_max, max(0.0, value))
        return top + plot_height * (1 - clipped / y_max)

    step = plot_width / len(stats)
    box_width = min(54.0, step * 0.55)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<style>text{font-family:Arial,sans-serif;fill:#172033}.grid{stroke:#d9dee8;stroke-width:1}"
        ".axis{stroke:#596579;stroke-width:1.5}.box{fill:#cfe2ff;stroke:#285ea8;stroke-width:2}"
        ".median{stroke:#d94841;stroke-width:3}.whisker{stroke:#285ea8;stroke-width:1.5}"
        ".outlier{fill:#f59f00;stroke:#a15c00;stroke-width:1}"
        ".clipped{fill:#7b2cbf;stroke:#4c1d6f;stroke-width:1}</style>",
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-size="22" '
        f'font-weight="bold">{escape(title)}</text>',
    ]
    for index in range(6):
        value = y_max * index / 5
        yy = y(value)
        parts.extend(
            [
                f'<line class="grid" x1="{left}" y1="{yy:.2f}" x2="{width - right}" y2="{yy:.2f}"/>',
                f'<text x="{left - 12}" y="{yy + 4:.2f}" text-anchor="end" font-size="12">{value:.0f}</text>',
            ]
        )
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}"/>',
            f'<text transform="translate(25 {top + plot_height / 2:.1f}) rotate(-90)" '
            f'text-anchor="middle" font-size="15">{escape(y_label)}</text>',
        ]
    )

    for index, (label, item) in enumerate(stats):
        x = left + step * (index + 0.5)
        q3_y, q1_y = y(item.q3), y(item.q1)
        parts.extend(
            [
                f'<g class="boxplot" data-label="{escape(label)}" '
                f'data-median-ms="{item.median:.6f}" data-count="{item.count}">',
                f'<line class="whisker" x1="{x:.2f}" y1="{y(item.whisker_high):.2f}" '
                f'x2="{x:.2f}" y2="{q3_y:.2f}"/>',
                f'<line class="whisker" x1="{x:.2f}" y1="{q1_y:.2f}" '
                f'x2="{x:.2f}" y2="{y(item.whisker_low):.2f}"/>',
                f'<line class="whisker" x1="{x - box_width / 3:.2f}" y1="{y(item.whisker_high):.2f}" '
                f'x2="{x + box_width / 3:.2f}" y2="{y(item.whisker_high):.2f}"/>',
                f'<line class="whisker" x1="{x - box_width / 3:.2f}" y1="{y(item.whisker_low):.2f}" '
                f'x2="{x + box_width / 3:.2f}" y2="{y(item.whisker_low):.2f}"/>',
                f'<rect class="box" x="{x - box_width / 2:.2f}" y="{q3_y:.2f}" width="{box_width:.2f}" '
                f'height="{max(1.0, q1_y - q3_y):.2f}"/>',
                f'<line class="median" x1="{x - box_width / 2:.2f}" y1="{y(item.median):.2f}" '
                f'x2="{x + box_width / 2:.2f}" y2="{y(item.median):.2f}"/>',
            ]
        )
        parts.extend(
            f'<circle class="outlier" cx="{x:.2f}" cy="{y(value):.2f}" r="3"/>'
            for value in item.outliers
        )
        clipped_count = sum(float(value) > y_max for value in series[label])
        if clipped_count:
            parts.extend(
                [
                    f'<polygon class="clipped" points="{x - 5:.2f},{top + 8:.2f} '
                    f'{x + 5:.2f},{top + 8:.2f} {x:.2f},{top:.2f}"/>',
                    f'<text x="{x + 8:.2f}" y="{top + 10:.2f}" font-size="10" '
                    f'fill="#4c1d6f">×{clipped_count}</text>',
                ]
            )
        parts.extend(
            [
                f'<text transform="translate({x:.2f} {top + plot_height + 18}) rotate(35)" '
                f'text-anchor="start" font-size="12">{escape(label)}</text>',
                "</g>",
            ]
        )
    parts.extend(
        [
            f'<text x="{width - right}" y="{height - 18}" text-anchor="end" font-size="11" '
            'fill="#596579">Box: p25–p75 · red: p50 · whiskers: 1.5×IQR · dots: outliers · triangles: values above axis limit</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return [label for label, _ in stats]


def write_latency_boxplots(
    output_dir: Path,
    samples: Sequence[Any],
    *,
    group_label: Callable[[Any], str],
    title_prefix: str,
    filename_prefix: str = "",
) -> dict[str, Path]:
    """Generate TTFB, initial-silence and TTFA plots from raw successful samples."""

    paths: dict[str, Path] = {}
    for metric, (field, label) in LATENCY_METRICS.items():
        grouped: dict[str, list[float]] = defaultdict(list)
        for sample in samples:
            status = sample.get("status") if isinstance(sample, dict) else sample.status
            value = sample.get(field) if isinstance(sample, dict) else getattr(sample, field)
            if status == "completed" and value is not None:
                grouped[group_label(sample)].append(float(value))
        if not grouped:
            continue
        path = output_dir / f"{filename_prefix}{metric}.svg"
        write_boxplot_svg(
            path,
            grouped,
            title=f"{title_prefix} {label}",
            y_label=f"{label} (ms)",
            y_max_ms=LATENCY_Y_MAX_MS[metric],
        )
        paths[metric] = path
    return paths

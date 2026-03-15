from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils.config import get_path_setting, load_path_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS_DIR = REPO_ROOT / "assets"


def parse_args() -> argparse.Namespace:
    path_config, _ = load_path_config()
    parser = argparse.ArgumentParser(description="Generate SVG summaries for LoRA evaluation results.")
    parser.add_argument(
        "--base-metrics",
        type=Path,
        default=None,
        help="Path to base model evaluation metrics JSON.",
    )
    parser.add_argument(
        "--adapter-metrics",
        type=Path,
        default=None,
        help="Path to LoRA adapter evaluation metrics JSON.",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help="Directory for generated SVGs and summary JSON.",
    )
    args = parser.parse_args()
    eval_dir = get_path_setting(path_config, "sft_eval_dir")
    if args.base_metrics is None:
        args.base_metrics = eval_dir / "base_test_metrics.json"
    else:
        args.base_metrics = args.base_metrics.expanduser().resolve()
    if args.adapter_metrics is None:
        args.adapter_metrics = eval_dir / "adapter_test_metrics.json"
    else:
        args.adapter_metrics = args.adapter_metrics.expanduser().resolve()
    args.assets_dir = args.assets_dir.expanduser().resolve()
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def percent(value: float) -> float:
    return round(value * 100.0, 1)


def build_summary(base_payload: dict[str, Any], adapter_payload: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    base_rows = base_payload["predictions"]
    adapter_rows = adapter_payload["predictions"]
    if len(base_rows) != len(adapter_rows):
        raise ValueError("Base and adapter prediction files contain different numbers of examples.")

    exact_improved = 0
    exact_regressed = 0
    relaxed_improved = 0
    relaxed_regressed = 0
    similarity_up = 0
    similarity_down = 0
    similarity_equal = 0
    deltas: list[float] = []

    for base_row, adapter_row in zip(base_rows, adapter_rows):
        if base_row["query"] != adapter_row["query"]:
            raise ValueError("Prediction rows do not align between base and adapter metrics.")

        if bool(adapter_row["exact_match"]) and not bool(base_row["exact_match"]):
            exact_improved += 1
        elif bool(base_row["exact_match"]) and not bool(adapter_row["exact_match"]):
            exact_regressed += 1

        if bool(adapter_row["relaxed_match"]) and not bool(base_row["relaxed_match"]):
            relaxed_improved += 1
        elif bool(base_row["relaxed_match"]) and not bool(adapter_row["relaxed_match"]):
            relaxed_regressed += 1

        delta = float(adapter_row["similarity_score"]) - float(base_row["similarity_score"])
        deltas.append(delta)
        if delta > 1e-12:
            similarity_up += 1
        elif delta < -1e-12:
            similarity_down += 1
        else:
            similarity_equal += 1

    total = len(deltas)
    summary = {
        "num_examples": total,
        "metrics": {
            "base": {
                "exact_match": percent(base_payload["exact_match"]),
                "relaxed_match": percent(base_payload["relaxed_match"]),
                "avg_similarity": percent(base_payload["avg_similarity"]),
            },
            "sft": {
                "exact_match": percent(adapter_payload["exact_match"]),
                "relaxed_match": percent(adapter_payload["relaxed_match"]),
                "avg_similarity": percent(adapter_payload["avg_similarity"]),
            },
        },
        "diff": {
            "exact": {
                "improved": exact_improved,
                "regressed": exact_regressed,
                "same": total - exact_improved - exact_regressed,
            },
            "relaxed": {
                "improved": relaxed_improved,
                "regressed": relaxed_regressed,
                "same": total - relaxed_improved - relaxed_regressed,
            },
            "similarity": {
                "improved": similarity_up,
                "regressed": similarity_down,
                "same": similarity_equal,
                "avg_delta": round(sum(deltas) / total, 4),
            },
        },
    }
    return summary, deltas


def histogram(values: list[float], bins: int = 17) -> tuple[list[int], list[float]]:
    lower = -1.0
    upper = 1.0
    step = (upper - lower) / bins
    counts = [0 for _ in range(bins)]
    edges = [lower + (step * i) for i in range(bins + 1)]
    for value in values:
        clipped = clamp(value, lower, upper)
        index = int((clipped - lower) / step)
        if index == bins:
            index -= 1
        counts[index] += 1
    return counts, edges


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" fill="none">'
    )


def make_overview_svg(summary: dict[str, Any]) -> str:
    width = 1280
    height = 720
    metrics = summary["metrics"]
    metric_names = ["exact_match", "relaxed_match", "avg_similarity"]
    labels = {
        "exact_match": "Exact Match",
        "relaxed_match": "Relaxed Match",
        "avg_similarity": "Avg Similarity",
    }
    base_color = "#B8C4D6"
    sft_color = "#1677FF"
    card_fill = "#F7F9FC"
    lines = [
        svg_header(width, height),
        "<defs>",
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#F8FBFF"/>',
        '<stop offset="100%" stop-color="#EDF3FF"/>',
        "</linearGradient>",
        "</defs>",
        f'<rect width="{width}" height="{height}" rx="28" fill="url(#bg)"/>',
        '<text x="64" y="84" fill="#0F172A" font-size="36" font-family="Arial, Helvetica, sans-serif" font-weight="700">ChartQA LoRA Evaluation</text>',
        '<text x="64" y="122" fill="#475569" font-size="18" font-family="Arial, Helvetica, sans-serif">Base vs SFT metrics on the ChartQA test split</text>',
    ]

    cards = [
        ("Examples", str(summary["num_examples"])),
        ("Exact Gain", f'+{summary["diff"]["exact"]["improved"]} / -{summary["diff"]["exact"]["regressed"]}'),
        ("Relaxed Gain", f'+{summary["diff"]["relaxed"]["improved"]} / -{summary["diff"]["relaxed"]["regressed"]}'),
        ("Avg Delta", f'{summary["diff"]["similarity"]["avg_delta"]:+.4f}'),
    ]
    for index, (title, value) in enumerate(cards):
        x = 64 + (index * 290)
        lines.extend(
            [
                f'<rect x="{x}" y="156" width="258" height="106" rx="24" fill="{card_fill}"/>',
                f'<text x="{x + 24}" y="196" fill="#64748B" font-size="16" font-family="Arial, Helvetica, sans-serif">{title}</text>',
                f'<text x="{x + 24}" y="235" fill="#0F172A" font-size="30" font-family="Arial, Helvetica, sans-serif" font-weight="700">{value}</text>',
            ]
        )

    chart_left = 84
    chart_top = 338
    chart_height = 280
    chart_width = 1100
    baseline_y = chart_top + chart_height
    group_spacing = chart_width / len(metric_names)
    bar_width = 92

    for tick in range(0, 101, 20):
        y = baseline_y - (chart_height * tick / 100.0)
        lines.append(f'<line x1="{chart_left}" y1="{y:.1f}" x2="{chart_left + chart_width}" y2="{y:.1f}" stroke="#D8E2F0" stroke-width="1"/>')
        lines.append(
            f'<text x="{chart_left - 16}" y="{y + 6:.1f}" text-anchor="end" fill="#64748B" font-size="14" font-family="Arial, Helvetica, sans-serif">{tick}%</text>'
        )

    for index, metric_name in enumerate(metric_names):
        group_x = chart_left + (group_spacing * index) + 90
        base_value = float(metrics["base"][metric_name])
        sft_value = float(metrics["sft"][metric_name])
        base_height = chart_height * base_value / 100.0
        sft_height = chart_height * sft_value / 100.0

        lines.extend(
            [
                f'<rect x="{group_x}" y="{baseline_y - base_height:.1f}" width="{bar_width}" height="{base_height:.1f}" rx="22" fill="{base_color}"/>',
                f'<rect x="{group_x + 122}" y="{baseline_y - sft_height:.1f}" width="{bar_width}" height="{sft_height:.1f}" rx="22" fill="{sft_color}"/>',
                f'<text x="{group_x + 46}" y="{baseline_y - base_height - 14:.1f}" text-anchor="middle" fill="#334155" font-size="16" font-family="Arial, Helvetica, sans-serif">{base_value:.1f}%</text>',
                f'<text x="{group_x + 168}" y="{baseline_y - sft_height - 14:.1f}" text-anchor="middle" fill="#0F172A" font-size="16" font-family="Arial, Helvetica, sans-serif" font-weight="700">{sft_value:.1f}%</text>',
                f'<text x="{group_x + 107}" y="{baseline_y + 38}" text-anchor="middle" fill="#334155" font-size="18" font-family="Arial, Helvetica, sans-serif">{labels[metric_name]}</text>',
                f'<text x="{group_x + 46}" y="{baseline_y + 68}" text-anchor="middle" fill="#64748B" font-size="14" font-family="Arial, Helvetica, sans-serif">Base</text>',
                f'<text x="{group_x + 168}" y="{baseline_y + 68}" text-anchor="middle" fill="#64748B" font-size="14" font-family="Arial, Helvetica, sans-serif">SFT</text>',
            ]
        )

    lines.extend(
        [
            '<rect x="944" y="58" width="18" height="18" rx="9" fill="#B8C4D6"/>',
            '<text x="972" y="73" fill="#334155" font-size="16" font-family="Arial, Helvetica, sans-serif">Base</text>',
            '<rect x="1050" y="58" width="18" height="18" rx="9" fill="#1677FF"/>',
            '<text x="1078" y="73" fill="#334155" font-size="16" font-family="Arial, Helvetica, sans-serif">SFT</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines)


def make_delta_svg(summary: dict[str, Any], deltas: list[float]) -> str:
    width = 1280
    height = 760
    card_fill = "#F8FAFC"
    improved_color = "#16A34A"
    regressed_color = "#DC2626"
    same_color = "#94A3B8"
    counts, edges = histogram(deltas)
    max_count = max(counts) if counts else 1
    lines = [
        svg_header(width, height),
        "<defs>",
        '<linearGradient id="bg2" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0%" stop-color="#FFFDF8"/>',
        '<stop offset="100%" stop-color="#FFF6EB"/>',
        "</linearGradient>",
        "</defs>",
        f'<rect width="{width}" height="{height}" rx="28" fill="url(#bg2)"/>',
        '<text x="64" y="84" fill="#111827" font-size="36" font-family="Arial, Helvetica, sans-serif" font-weight="700">Sample-Level Change Breakdown</text>',
        '<text x="64" y="122" fill="#4B5563" font-size="18" font-family="Arial, Helvetica, sans-serif">Improvement and regression counts plus similarity delta distribution</text>',
    ]

    breakdowns = [
        ("Exact Match", summary["diff"]["exact"]),
        ("Relaxed Match", summary["diff"]["relaxed"]),
        ("Similarity", summary["diff"]["similarity"]),
    ]
    for index, (title, payload) in enumerate(breakdowns):
        x = 64 + (index * 384)
        lines.append(f'<rect x="{x}" y="164" width="352" height="186" rx="24" fill="{card_fill}"/>')
        lines.append(f'<text x="{x + 24}" y="202" fill="#111827" font-size="24" font-family="Arial, Helvetica, sans-serif" font-weight="700">{title}</text>')
        rows = [
            ("Improved", payload["improved"], improved_color),
            ("Regressed", payload["regressed"], regressed_color),
            ("Same", payload["same"], same_color),
        ]
        if title == "Similarity":
            rows[0] = ("Up", payload["improved"], improved_color)
            rows[1] = ("Down", payload["regressed"], regressed_color)
        for row_index, (label, value, color) in enumerate(rows):
            y = 238 + (row_index * 38)
            lines.extend(
                [
                    f'<circle cx="{x + 28}" cy="{y - 6}" r="6" fill="{color}"/>',
                    f'<text x="{x + 44}" y="{y}" fill="#475569" font-size="16" font-family="Arial, Helvetica, sans-serif">{label}</text>',
                    f'<text x="{x + 326}" y="{y}" text-anchor="end" fill="#111827" font-size="18" font-family="Arial, Helvetica, sans-serif" font-weight="700">{value}</text>',
                ]
            )
        if title == "Similarity":
            lines.append(
                f'<text x="{x + 24}" y="330" fill="#64748B" font-size="15" font-family="Arial, Helvetica, sans-serif">Mean delta {payload["avg_delta"]:+.4f}</text>'
            )

    chart_left = 90
    chart_top = 430
    chart_width = 1100
    chart_height = 240
    base_y = chart_top + chart_height
    zero_x = chart_left + (chart_width / 2.0)
    bin_width = chart_width / len(counts)

    lines.append(f'<rect x="64" y="388" width="1152" height="312" rx="28" fill="white" fill-opacity="0.7"/>')
    lines.append(f'<line x1="{chart_left}" y1="{base_y}" x2="{chart_left + chart_width}" y2="{base_y}" stroke="#CBD5E1" stroke-width="2"/>')
    lines.append(f'<line x1="{zero_x:.1f}" y1="{chart_top - 12}" x2="{zero_x:.1f}" y2="{base_y}" stroke="#94A3B8" stroke-dasharray="8 8" stroke-width="2"/>')

    for tick in range(0, max_count + 1, max(1, max_count // 4 or 1)):
        y = base_y - (chart_height * tick / max_count if max_count else 0)
        lines.append(f'<line x1="{chart_left}" y1="{y:.1f}" x2="{chart_left + chart_width}" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        lines.append(
            f'<text x="{chart_left - 14}" y="{y + 5:.1f}" text-anchor="end" fill="#64748B" font-size="13" font-family="Arial, Helvetica, sans-serif">{tick}</text>'
        )

    for index, count in enumerate(counts):
        x = chart_left + (index * bin_width) + 4
        height_value = chart_height * count / max_count if max_count else 0
        y = base_y - height_value
        midpoint = (edges[index] + edges[index + 1]) / 2.0
        color = improved_color if midpoint >= 0 else regressed_color
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bin_width - 8:.1f}" height="{height_value:.1f}" rx="10" fill="{color}" fill-opacity="0.85"/>'
        )

    labels = [edges[0], -0.5, 0.0, 0.5, edges[-1]]
    for label in labels:
        x = chart_left + ((label + 1.0) / 2.0) * chart_width
        lines.append(
            f'<text x="{x:.1f}" y="{base_y + 28}" text-anchor="middle" fill="#64748B" font-size="13" font-family="Arial, Helvetica, sans-serif">{label:+.1f}</text>'
        )

    lines.extend(
        [
            '<text x="90" y="416" fill="#111827" font-size="22" font-family="Arial, Helvetica, sans-serif" font-weight="700">Similarity delta histogram</text>',
            '<text x="90" y="446" fill="#64748B" font-size="15" font-family="Arial, Helvetica, sans-serif">Delta = SFT similarity score - Base similarity score</text>',
            '<text x="1146" y="416" fill="#16A34A" font-size="15" font-family="Arial, Helvetica, sans-serif" text-anchor="end">positive deltas</text>',
            '<text x="1146" y="438" fill="#DC2626" font-size="15" font-family="Arial, Helvetica, sans-serif" text-anchor="end">negative deltas</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    base_payload = load_json(args.base_metrics)
    adapter_payload = load_json(args.adapter_metrics)
    summary, deltas = build_summary(base_payload, adapter_payload)

    summary_path = args.assets_dir / "lora_metrics_summary.json"
    overview_path = args.assets_dir / "lora_metrics_overview.svg"
    delta_path = args.assets_dir / "lora_metrics_delta.svg"

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    overview_path.write_text(make_overview_svg(summary), encoding="utf-8")
    delta_path.write_text(make_delta_svg(summary, deltas), encoding="utf-8")


if __name__ == "__main__":
    main()

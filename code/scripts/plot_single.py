"""Generate the legacy single-intersection result figure from JSON metrics."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FINAL_ASSETS = ROOT / "code" / "visualization" / "final_results"
DEFAULT_SUMMARY = FINAL_ASSETS / "evaluation_summary_single.json"
DEFAULT_OUTPUT = FINAL_ASSETS / "single_intersection_result.svg"


def improvement_percent(baseline, rl):
    if baseline == 0:
        return 0.0
    return ((baseline - rl) / baseline) * 100.0


def load_improvements(summary_path):
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    baseline = data["baseline"]
    rl = data["rl"]
    return [
        (
            "Avg Queue / Step",
            improvement_percent(
                baseline["avg_queue_per_step"], rl["avg_queue_per_step"]
            ),
        ),
        (
            "Avg Trip Duration",
            improvement_percent(baseline["avg_trip_duration"], rl["avg_trip_duration"]),
        ),
        (
            "Avg Waiting Time",
            improvement_percent(baseline["avg_waiting_time"], rl["avg_waiting_time"]),
        ),
        (
            "Avg Time Loss",
            improvement_percent(baseline["avg_time_loss"], rl["avg_time_loss"]),
        ),
        (
            "Max Waiting Time",
            improvement_percent(baseline["max_waiting_time"], rl["max_waiting_time"]),
        ),
    ]


def write_svg(output_path, improvements):
    width, height = 980, 450
    left, right = 210, 920
    chart_top, chart_bottom = 80, 390
    max_axis = 60
    row_y = [105, 165, 225, 285, 345]
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '  <rect width="100%" height="100%" fill="#ffffff"/>',
        "  <style>",
        "    .title { font: 700 26px Arial, sans-serif; fill: #111827; }",
        "    .label { font: 700 16px Arial, sans-serif; fill: #111827; }",
        "    .tick { font: 400 15px Arial, sans-serif; fill: #374151; }",
        "    .value { font: 700 17px Arial, sans-serif; fill: #111827; }",
        "  </style>",
        "",
        f'  <text x="{width / 2:.0f}" y="44" text-anchor="middle" '
        'class="title">Single-Intersection RL Improvements Over Fixed-Time</text>',
        "",
        f'  <line x1="{left}" y1="{chart_top}" x2="{left}" '
        f'y2="{chart_bottom}" stroke="#111827" stroke-width="1.4"/>',
        f'  <line x1="{left}" y1="{chart_bottom}" x2="{right}" '
        f'y2="{chart_bottom}" stroke="#111827" stroke-width="1.4"/>',
        "",
    ]

    for tick in range(0, max_axis + 1, 10):
        x = left + ((right - left) * tick / max_axis)
        body.append(
            f'  <line x1="{x:g}" y1="{chart_top}" x2="{x:g}" '
            f'y2="{chart_bottom}" stroke="#E5E7EB"/>'
        )
        body.append(
            f'  <text x="{x:g}" y="412" text-anchor="middle" '
            f'class="tick">{tick}%</text>'
        )

    body.append("")
    for (label, value), y in zip(improvements, row_y):
        bar_width = min(max(value, 0.0), max_axis) / max_axis * (right - left)
        value_x = left + bar_width + 8
        anchor = "start"
        if value > max_axis - 3:
            value_x = right + 8
            anchor = "end"
        body.extend(
            [
                f'  <text x="194" y="{y + 18}" text-anchor="end" '
                f'class="label">{label}</text>',
                f'  <rect x="{left}" y="{y}" width="{bar_width:.1f}" '
                'height="28" fill="#377684" rx="3"/>',
                f'  <text x="{value_x:.1f}" y="{y + 19}" '
                f'text-anchor="{anchor}" class="value">+{value:.1f}%</text>',
                "",
            ]
        )

    body.append("</svg>")
    Path(output_path).write_text("\n".join(body) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot legacy single-intersection results."
    )
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_svg(output_path, load_improvements(args.summary_json))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

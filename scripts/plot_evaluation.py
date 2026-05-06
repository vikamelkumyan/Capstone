"""Generate SVG evaluation plots from the final evaluation summary JSON."""

import argparse
import json
from pathlib import Path


COLORS = {
    "fixed_time": "#8A95A5",
    "max_pressure": "#5B78A7",
    "rl": "#377684",
    "good": "#377684",
    "bad": "#B91C1C",
    "neutral": "#6B7280",
}


def svg_text(x, y, text, size=12, anchor="start", weight="400", fill="#111827"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{text}</text>'
    )


def write_svg(path, width, height, body):
    content = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#FFFFFF"/>',
            *body,
            "</svg>",
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def controller_label(controller):
    return {
        "fixed_time": "Fixed-Time",
        "max_pressure": "MaxPressure",
        "rl": "RL",
    }[controller]


def plot_wait_by_scenario(data, output):
    scenarios = ["corridor_stress", "evening_rush", "off_peak"]
    scenario_labels = {
        "corridor_stress": "stress test",
        "evening_rush": "rush hour",
        "off_peak": "off peak",
    }
    controllers = ["fixed_time", "max_pressure", "rl"]
    means = {
        (row["scenario"], row["controller"]): row["avg_wait"]
        for row in data["per_scenario_means"]
    }

    width, height = 980, 560
    margin_left, margin_right = 90, 30
    margin_top, margin_bottom = 70, 90
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    max_value = max(means.values())
    y_max = 2200
    if max_value > y_max:
        y_max = ((int(max_value) // 500) + 1) * 500

    body = [
        svg_text(
            width / 2,
            38,
            "Multi-Intersection Average Vehicle Waiting Time by Scenario",
            22,
            "middle",
            "700",
        ),
    ]

    # Axes and grid.
    for tick in range(0, y_max + 1, 500):
        y = margin_top + chart_h - (tick / y_max) * chart_h
        body.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#E5E7EB"/>'
        )
        body.append(
            svg_text(margin_left - 12, y + 5, str(tick), 13, "end", fill="#374151")
        )
    body.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#111827"/>'
    )
    body.append(
        f'<line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{width - margin_right}" y2="{margin_top + chart_h}" stroke="#111827"/>'
    )

    group_w = chart_w / len(scenarios)
    bar_w = 42
    for i, scenario in enumerate(scenarios):
        group_x = margin_left + i * group_w
        center = group_x + group_w / 2
        label = scenario_labels[scenario]
        body.append(
            svg_text(center, margin_top + chart_h + 38, label, 15, "middle", "700")
        )

        for j, controller in enumerate(controllers):
            value = means[(scenario, controller)]
            bar_h = (value / y_max) * chart_h
            x = center - (1.5 * bar_w) + j * (bar_w + 8)
            y = margin_top + chart_h - bar_h
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{COLORS[controller]}" rx="3"/>'
            )
            body.append(
                svg_text(
                    x + bar_w / 2,
                    y - 7,
                    f"{value:.0f}",
                    14,
                    "middle",
                    "700",
                    "#111827",
                )
            )

    legend_y = height - 28
    legend_x = margin_left
    for controller in controllers:
        body.append(
            f'<rect x="{legend_x}" y="{legend_y - 11}" width="14" height="14" fill="{COLORS[controller]}" rx="2"/>'
        )
        body.append(
            svg_text(legend_x + 20, legend_y + 1, controller_label(controller), 14)
        )
        legend_x += 150

    body.append(svg_text(22, margin_top + chart_h / 2, "Seconds", 14, "middle", "700"))
    write_svg(output, width, height, body)


def plot_overall_delta(data, output):
    ordered = [
        ("avg_queue_per_step", "Avg Queue / Step"),
        ("avg_trip_duration", "Avg Trip Duration"),
        ("avg_waiting_time", "Avg Waiting Time"),
        ("avg_time_loss", "Avg Time Loss"),
        ("max_waiting_time", "Max Waiting Time"),
    ]
    values = [
        (label, data["overall_mean_std"][key]["delta_rl_vs_fixed_percent"])
        for key, label in ordered
    ]

    width, height = 980, 520
    margin_left, margin_right = 210, 70
    margin_top, margin_bottom = 70, 50
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    axis_min, axis_max = 0, 20
    zero_x = margin_left

    body = [
        svg_text(
            width / 2,
            34,
            "Multi-Intersection Delay Improvements Over Fixed-Time",
            22,
            "middle",
            "700",
        ),
        svg_text(
            width / 2,
            56,
            "Positive values mean the RL controller reduced queueing, delay, or waiting time.",
            13,
            "middle",
            "400",
            "#374151",
        ),
    ]

    for tick in range(axis_min, axis_max + 1, 5):
        x = margin_left + ((tick - axis_min) / (axis_max - axis_min)) * chart_w
        body.append(
            f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + chart_h}" stroke="#E5E7EB"/>'
        )
        body.append(
            svg_text(
                x, margin_top + chart_h + 24, f"{tick}%", 13, "middle", fill="#374151"
            )
        )
    body.append(
        f'<line x1="{zero_x:.1f}" y1="{margin_top}" x2="{zero_x:.1f}" y2="{margin_top + chart_h}" stroke="#111827"/>'
    )

    row_h = chart_h / len(values)
    for i, (label, value) in enumerate(values):
        y = margin_top + i * row_h + row_h * 0.25
        bar_h = row_h * 0.5
        x_value = margin_left + ((value - axis_min) / (axis_max - axis_min)) * chart_w
        x_value = max(margin_left, min(width - margin_right, x_value))
        x = zero_x
        bar_w = abs(x_value - zero_x)
        color = COLORS["good"] if value > 0 else COLORS["bad"]
        body.append(
            svg_text(margin_left - 16, y + bar_h * 0.68, label, 14, "end", "700")
        )
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>'
        )
        label_x = x_value + 8
        anchor = "start"
        body.append(
            svg_text(label_x, y + bar_h * 0.68, f"{value:+.1f}%", 15, anchor, "700")
        )

    write_svg(output, width, height, body)


def plot_policy_diagnostics(data, output):
    rows = data["policy_diagnostics"]
    width, height = 980, 460
    margin_left, margin_right = 220, 70
    margin_top, margin_bottom = 70, 45
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    body = [
        svg_text(
            width / 2,
            32,
            "Multi-Intersection RL Action Stability by Traffic Light",
            22,
            "middle",
            "700",
        ),
        svg_text(
            width / 2,
            54,
            "Dominant action share; labels show total switches. Blocked switches were zero for every TLS.",
            13,
            "middle",
            "400",
            "#374151",
        ),
    ]

    for tick in range(0, 101, 20):
        x = margin_left + (tick / 100) * chart_w
        body.append(
            f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + chart_h}" stroke="#E5E7EB"/>'
        )
        body.append(
            svg_text(
                x, margin_top + chart_h + 24, f"{tick}%", 13, "middle", fill="#374151"
            )
        )

    row_h = chart_h / len(rows)
    for i, row in enumerate(rows):
        y = margin_top + i * row_h + row_h * 0.22
        bar_h = row_h * 0.52
        bar_w = (row["dominance_percent"] / 100) * chart_w
        body.append(
            svg_text(
                margin_left - 16,
                y + bar_h * 0.68,
                row["tls"].replace("Komitas-", ""),
                14,
                "end",
                "700",
            )
        )
        body.append(
            f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{COLORS["rl"]}" rx="3"/>'
        )
        body.append(
            svg_text(
                margin_left + bar_w + 8,
                y + bar_h * 0.68,
                f'{row["dominance_percent"]:.1f}% | {row["switches"]} switches',
                14,
                "start",
                "600",
            )
        )

    write_svg(output, width, height, body)


def main():
    parser = argparse.ArgumentParser(description="Generate final evaluation SVG plots.")
    parser.add_argument(
        "--summary-json", default="docs/assets/final_results/eval_ep075_summary.json"
    )
    parser.add_argument("--output-dir", default="docs/assets/final_results")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    plot_wait_by_scenario(data, output_dir / "evaluation_wait_by_scenario.svg")
    plot_overall_delta(data, output_dir / "evaluation_delta_vs_fixed.svg")
    plot_policy_diagnostics(data, output_dir / "policy_diagnostics.svg")


if __name__ == "__main__":
    main()

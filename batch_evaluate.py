import argparse
import json
import statistics
from pathlib import Path

from evaluate import evaluate_pair
from scripts.generate_traffic import SCENARIOS, generate_route_file


def format_metric(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


METRIC_KEYS = [
    "steps",
    "arrived_vehicles",
    "avg_queue_per_step",
    "avg_lane_wait_per_step",
    "avg_trip_duration",
    "avg_waiting_time",
    "avg_time_loss",
    "max_waiting_time",
]


def average_metrics(results, controller_key):
    count = max(len(results), 1)
    return {
        key: sum(item[controller_key][key] for item in results) / count
        for key in METRIC_KEYS
    }


def std_metrics(results, controller_key):
    if len(results) < 2:
        return {key: 0.0 for key in METRIC_KEYS}
    return {
        key: statistics.stdev(item[controller_key][key] for item in results)
        for key in METRIC_KEYS
    }


def print_aggregate_table(
    baseline_avg, rl_avg, baseline_std=None, rl_std=None, mp_avg=None, mp_std=None
):
    def fmt(mean, std=None):
        if isinstance(mean, float):
            if std is not None and std > 0:
                return f"{mean:.2f} ± {std:.2f}"
            return f"{mean:.2f}"
        return str(mean)

    def pct(b, val, higher_is_better=False):
        if b == 0:
            return "n/a"
        delta = (
            ((val - b) / abs(b) * 100)
            if higher_is_better
            else ((b - val) / abs(b) * 100)
        )
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.1f}%"

    metric_specs = [
        ("Steps", "steps", False),
        ("Vehicles Arrived", "arrived_vehicles", True),
        ("Avg Queue / Step", "avg_queue_per_step", False),
        ("Avg Lane Wait / Step", "avg_lane_wait_per_step", False),
        ("Avg Trip Duration", "avg_trip_duration", False),
        ("Avg Waiting Time", "avg_waiting_time", False),
        ("Avg Time Loss", "avg_time_loss", False),
        ("Max Waiting Time", "max_waiting_time", False),
    ]

    has_mp = mp_avg is not None
    if has_mp:
        headers = ("Metric", "Fixed-Time", "MaxPressure", "RL", "Δ% (RL vs Fixed)")
    else:
        headers = ("Metric", "Fixed-Time", "RL", "Δ% (RL vs Fixed)")

    rows = []
    for label, key, higher in metric_specs:
        b_std = baseline_std[key] if baseline_std else None
        r_std = rl_std[key] if rl_std else None
        if has_mp:
            m_std = mp_std[key] if mp_std else None
            rows.append(
                (
                    label,
                    fmt(baseline_avg[key], b_std),
                    fmt(mp_avg[key], m_std),
                    fmt(rl_avg[key], r_std),
                    pct(baseline_avg[key], rl_avg[key], higher_is_better=higher),
                )
            )
        else:
            rows.append(
                (
                    label,
                    fmt(baseline_avg[key], b_std),
                    fmt(rl_avg[key], r_std),
                    pct(baseline_avg[key], rl_avg[key], higher_is_better=higher),
                )
            )

    n = len(headers)
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(n)]
    sep = "  "
    print(
        sep.join(
            f"{headers[i]:<{widths[i]}}" if i == 0 else f"{headers[i]:>{widths[i]}}"
            for i in range(n)
        )
    )
    print(sep.join("-" * w for w in widths))
    for row in rows:
        print(
            sep.join(
                f"{row[i]:<{widths[i]}}" if i == 0 else f"{row[i]:>{widths[i]}}"
                for i in range(n)
            )
        )


def print_per_route_summary(results):
    has_mp = results and "max_pressure" in results[0]
    if has_mp:
        headers = (
            "Scenario",
            "Seed",
            "Fixed Avg Wait",
            "MaxPres Avg Wait",
            "RL Avg Wait",
            "Fixed Max Wait",
            "RL Max Wait",
        )
        rows = [
            (
                item["scenario"],
                str(item["seed"]),
                f"{item['baseline']['avg_waiting_time']:.2f}",
                f"{item['max_pressure']['avg_waiting_time']:.2f}",
                f"{item['rl']['avg_waiting_time']:.2f}",
                f"{item['baseline']['max_waiting_time']:.2f}",
                f"{item['rl']['max_waiting_time']:.2f}",
            )
            for item in results
        ]
    else:
        headers = (
            "Scenario",
            "Seed",
            "Fixed Avg Wait",
            "RL Avg Wait",
            "Fixed Max Wait",
            "RL Max Wait",
        )
        rows = [
            (
                item["scenario"],
                str(item["seed"]),
                f"{item['baseline']['avg_waiting_time']:.2f}",
                f"{item['rl']['avg_waiting_time']:.2f}",
                f"{item['baseline']['max_waiting_time']:.2f}",
                f"{item['rl']['max_waiting_time']:.2f}",
            )
            for item in results
        ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        if rows
        else len(headers[idx])
        for idx in range(len(headers))
    ]

    print(
        "  ".join(
            f"{header:<{widths[idx]}}" if idx < 2 else f"{header:>{widths[idx]}}"
            for idx, header in enumerate(headers)
        )
    )
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                f"{value:<{widths[idx]}}" if idx < 2 else f"{value:>{widths[idx]}}"
                for idx, value in enumerate(row)
            )
        )


def print_policy_diagnostics(results):
    """Summarize whether the RL policies collapsed to one dominant action."""
    if not results or "action_counts" not in results[0].get("rl", {}):
        return

    tls_action_counts = {}
    tls_switch_counts = {}
    tls_blocked_counts = {}

    for item in results:
        rl = item["rl"]
        for tls_id, counts in rl.get("action_counts", {}).items():
            tls_action_counts.setdefault(tls_id, {})
            for action_id, count in counts.items():
                tls_action_counts[tls_id][str(action_id)] = (
                    tls_action_counts[tls_id].get(str(action_id), 0) + count
                )
        for tls_id, count in rl.get("switch_counts", {}).items():
            tls_switch_counts[tls_id] = tls_switch_counts.get(tls_id, 0) + count
        for tls_id, count in rl.get("blocked_switch_counts", {}).items():
            tls_blocked_counts[tls_id] = tls_blocked_counts.get(tls_id, 0) + count

    headers = ("TLS", "Dominant Action", "Dominance", "Switches", "Blocked")
    rows = []
    for tls_id in sorted(tls_action_counts):
        counts = tls_action_counts[tls_id]
        total = sum(counts.values())
        dominant_action, dominant_count = max(counts.items(), key=lambda item: item[1])
        dominance = dominant_count / max(total, 1) * 100.0
        rows.append(
            (
                tls_id,
                dominant_action,
                f"{dominance:.1f}%",
                str(tls_switch_counts.get(tls_id, 0)),
                str(tls_blocked_counts.get(tls_id, 0)),
            )
        )

    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]
    print("  ".join(f"{headers[idx]:<{widths[idx]}}" for idx in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(f"{row[idx]:<{widths[idx]}}" for idx in range(len(row))))


def print_per_scenario_summary(results):
    """One-line-per-scenario breakdown so off-peak collapse cannot hide in averages.

    The previous aggregate-only view averaged across all scenarios; the catastrophic
    off-peak wait time (e.g. 596 s vs baseline 34 s) was diluted in the mean.
    """
    by_scenario = {}
    for r in results:
        by_scenario.setdefault(r["scenario"], []).append(r)

    print("Per-scenario controller comparison (mean across seeds):")
    headers = (
        "Scenario",
        "Controller",
        "Arrived",
        "Avg Q",
        "Avg Wait",
        "Trip Dur",
        "Switches/route",
    )
    rows = []
    for scenario in sorted(by_scenario):
        items = by_scenario[scenario]
        for label in ("baseline", "max_pressure", "rl"):
            if not items or label not in items[0]:
                continue
            n = len(items)
            arrived = sum(i[label]["arrived_vehicles"] for i in items) / n
            avg_q = sum(i[label]["avg_queue_per_step"] for i in items) / n
            wait = sum(i[label]["avg_waiting_time"] for i in items) / n
            trip = sum(i[label]["avg_trip_duration"] for i in items) / n
            sw = ""
            if label == "rl" and items and "switch_counts" in items[0]["rl"]:
                total_switches = (
                    sum(sum(i["rl"]["switch_counts"].values()) for i in items) / n
                )
                sw = f"{total_switches:.0f}"
            rows.append(
                (
                    scenario,
                    label,
                    f"{arrived:.0f}",
                    f"{avg_q:.2f}",
                    f"{wait:.1f}",
                    f"{trip:.1f}",
                    sw,
                )
            )

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    print(
        "  ".join(
            f"{headers[i]:<{widths[i]}}" if i < 2 else f"{headers[i]:>{widths[i]}}"
            for i in range(len(headers))
        )
    )
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(
            "  ".join(
                f"{row[i]:<{widths[i]}}" if i < 2 else f"{row[i]:>{widths[i]}}"
                for i in range(len(row))
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained RL controller across multiple generated traffic scenarios."
    )
    parser.add_argument(
        "--model-path",
        default="dqn_model.pth",
        help="Path to the trained model used for RL evaluation.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=sorted(SCENARIOS),
        help="Traffic scenarios to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[41, 42, 43],
        help="Random seeds used to generate route files.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3600,
        help="Seconds of demand to generate. Evaluation can run longer via --end-time.",
    )
    parser.add_argument(
        "--end-time",
        type=int,
        default=7200,
        help="Simulation seconds allowed for each generated route file.",
    )
    parser.add_argument(
        "--route-dir",
        default="runs/batch_eval_routes",
        help="Directory used for generated route files.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write the full batch evaluation summary as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    route_dir = Path(args.route_dir)
    route_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            route_file = route_dir / f"{scenario}_seed{seed}.rou.xml"
            generate_route_file(
                route_file,
                SCENARIOS[scenario],
                seed=seed,
                n_steps=args.steps,
            )

            print(f"Evaluating scenario={scenario} seed={seed} ...")
            pair = evaluate_pair(
                model_path=args.model_path,
                route_file=str(route_file),
                end_time=args.end_time,
            )
            results.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "route_file": str(route_file),
                    "baseline": pair["baseline"],
                    "max_pressure": pair["max_pressure"],
                    "rl": pair["rl"],
                }
            )

    baseline_avg = average_metrics(results, "baseline")
    rl_avg = average_metrics(results, "rl")
    baseline_std = std_metrics(results, "baseline")
    rl_std = std_metrics(results, "rl")
    has_mp = results and "max_pressure" in results[0]
    mp_avg = average_metrics(results, "max_pressure") if has_mp else None
    mp_std = std_metrics(results, "max_pressure") if has_mp else None

    print()
    print("Per-route summary:")
    print_per_route_summary(results)
    print()
    print_per_scenario_summary(results)
    print()
    print("Averaged comparison (mean ± std across scenarios/seeds):")
    print_aggregate_table(baseline_avg, rl_avg, baseline_std, rl_std, mp_avg, mp_std)
    print()
    print("RL policy diagnostics across evaluated routes:")
    print_policy_diagnostics(results)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_path": args.model_path,
            "scenarios": args.scenarios,
            "seeds": args.seeds,
            "steps": args.steps,
            "end_time": args.end_time,
            "results": results,
            "baseline_average": baseline_avg,
            "baseline_std": baseline_std,
            "max_pressure_average": mp_avg,
            "max_pressure_std": mp_std,
            "rl_average": rl_avg,
            "rl_std": rl_std,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"Detailed summary written to {summary_path}")


if __name__ == "__main__":
    main()

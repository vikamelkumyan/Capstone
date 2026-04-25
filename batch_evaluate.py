import argparse
import json
from pathlib import Path

from evaluate import evaluate_pair
from scripts.generate_traffic import SCENARIOS, generate_route_file


def format_metric(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def average_metrics(results, controller_key):
    metric_keys = [
        "steps",
        "arrived_vehicles",
        "avg_queue_per_step",
        "avg_lane_wait_per_step",
        "avg_trip_duration",
        "avg_waiting_time",
        "avg_time_loss",
        "max_waiting_time",
    ]
    count = max(len(results), 1)
    return {
        key: sum(item[controller_key][key] for item in results) / count
        for key in metric_keys
    }


def print_aggregate_table(baseline_avg, rl_avg):
    rows = [
        ("Steps", format_metric(baseline_avg["steps"]), format_metric(rl_avg["steps"])),
        (
            "Vehicles Arrived",
            format_metric(baseline_avg["arrived_vehicles"]),
            format_metric(rl_avg["arrived_vehicles"]),
        ),
        (
            "Avg Queue / Step",
            format_metric(baseline_avg["avg_queue_per_step"]),
            format_metric(rl_avg["avg_queue_per_step"]),
        ),
        (
            "Avg Lane Wait / Step",
            format_metric(baseline_avg["avg_lane_wait_per_step"]),
            format_metric(rl_avg["avg_lane_wait_per_step"]),
        ),
        (
            "Avg Trip Duration",
            format_metric(baseline_avg["avg_trip_duration"]),
            format_metric(rl_avg["avg_trip_duration"]),
        ),
        (
            "Avg Waiting Time",
            format_metric(baseline_avg["avg_waiting_time"]),
            format_metric(rl_avg["avg_waiting_time"]),
        ),
        (
            "Avg Time Loss",
            format_metric(baseline_avg["avg_time_loss"]),
            format_metric(rl_avg["avg_time_loss"]),
        ),
        (
            "Max Waiting Time",
            format_metric(baseline_avg["max_waiting_time"]),
            format_metric(rl_avg["max_waiting_time"]),
        ),
    ]

    headers = ("Metric", "Baseline Avg", "RL Avg")
    widths = [
        max(len(headers[0]), *(len(row[0]) for row in rows)),
        max(len(headers[1]), *(len(row[1]) for row in rows)),
        max(len(headers[2]), *(len(row[2]) for row in rows)),
    ]

    print(
        f"{headers[0]:<{widths[0]}}  {headers[1]:>{widths[1]}}  {headers[2]:>{widths[2]}}"
    )
    print(f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}")
    for metric, baseline_value, rl_value in rows:
        print(
            f"{metric:<{widths[0]}}  {baseline_value:>{widths[1]}}  {rl_value:>{widths[2]}}"
        )


def print_per_route_summary(results):
    headers = (
        "Scenario",
        "Seed",
        "Baseline Avg Wait",
        "RL Avg Wait",
        "Baseline Max Wait",
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
        default=7200,
        help="Simulation length used for generated evaluation traffic files.",
    )
    parser.add_argument(
        "--route-dir",
        default="batch_eval_routes",
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
            if not route_file.exists():
                generate_route_file(
                    route_file,
                    SCENARIOS[scenario],
                    seed=seed,
                    n_steps=args.steps,
                )

            print(f"Evaluating scenario={scenario} seed={seed} ...")
            pair = evaluate_pair(model_path=args.model_path, route_file=str(route_file))
            results.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "route_file": str(route_file),
                    "baseline": pair["baseline"],
                    "rl": pair["rl"],
                }
            )

    baseline_avg = average_metrics(results, "baseline")
    rl_avg = average_metrics(results, "rl")

    print()
    print("Per-route summary:")
    print_per_route_summary(results)
    print()
    print("Averaged comparison:")
    print_aggregate_table(baseline_avg, rl_avg)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_path": args.model_path,
            "scenarios": args.scenarios,
            "seeds": args.seeds,
            "steps": args.steps,
            "results": results,
            "baseline_average": baseline_avg,
            "rl_average": rl_avg,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"Detailed summary written to {summary_path}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from evaluate import evaluate_pair, run_default_controller
from scripts.generate_traffic import SCENARIOS, generate_route_file
from train import train


def checkpoint_sort_key(path):
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def improvement_ratio(baseline_value, rl_value):
    if baseline_value == 0:
        return 0.0
    return (baseline_value - rl_value) / baseline_value


def score_result(baseline, rl):
    """Score one RL result relative to baseline. Higher is better."""
    score = 0.0
    score += 2.5 * improvement_ratio(
        baseline["avg_waiting_time"], rl["avg_waiting_time"]
    )
    score += 2.0 * improvement_ratio(
        baseline["avg_lane_wait_per_step"], rl["avg_lane_wait_per_step"]
    )
    score += 1.5 * improvement_ratio(
        baseline["avg_queue_per_step"], rl["avg_queue_per_step"]
    )
    score += 1.0 * improvement_ratio(
        baseline["avg_trip_duration"], rl["avg_trip_duration"]
    )
    score += 1.0 * improvement_ratio(baseline["avg_time_loss"], rl["avg_time_loss"])
    score += 1.0 * improvement_ratio(
        baseline["max_waiting_time"], rl["max_waiting_time"]
    )

    throughput_gap = baseline["arrived_vehicles"] - rl["arrived_vehicles"]
    if throughput_gap > 0:
        score -= 5.0 * (throughput_gap / max(baseline["arrived_vehicles"], 1))

    return score


def build_eval_routes(output_dir, scenarios, seeds, steps):
    """Generate deterministic evaluation route files."""
    route_files = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        for seed in seeds:
            route_path = output_dir / f"{scenario}_seed{seed}.rou.xml"
            if not route_path.exists():
                generate_route_file(
                    route_path, SCENARIOS[scenario], seed=seed, n_steps=steps
                )
            route_files.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "route_file": str(route_path),
                }
            )

    return route_files


def evaluate_checkpoints(checkpoints, eval_routes):
    """Evaluate each checkpoint across all evaluation routes."""
    baseline_cache = {}
    all_results = []

    for route in eval_routes:
        baseline_cache[route["route_file"]] = run_default_controller(
            route_file=route["route_file"]
        )

    for checkpoint in checkpoints:
        per_route = []
        total_score = 0.0

        for route in eval_routes:
            baseline = baseline_cache[route["route_file"]]
            rl = evaluate_pair(
                model_path=str(checkpoint), route_file=route["route_file"]
            )["rl"]
            score = score_result(baseline, rl)
            total_score += score

            per_route.append(
                {
                    "scenario": route["scenario"],
                    "seed": route["seed"],
                    "baseline": baseline,
                    "rl": rl,
                    "score": score,
                }
            )

        avg_score = total_score / max(len(eval_routes), 1)
        all_results.append(
            {
                "checkpoint": str(checkpoint),
                "avg_score": avg_score,
                "per_route": per_route,
            }
        )

    return sorted(all_results, key=lambda item: item["avg_score"], reverse=True)


def format_metric(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_markdown_summary(ranked_results, output_path, top_k=5):
    """Write a readable Markdown summary of the best checkpoints."""
    top_results = ranked_results[:top_k]

    lines = [
        "# Checkpoint Summary",
        "",
        f"Top {len(top_results)} checkpoints ranked by average cross-scenario score.",
        "",
        "## Leaderboard",
        "",
        "| Rank | Checkpoint | Avg Score |",
        "|---|---|---:|",
    ]

    for idx, item in enumerate(top_results, start=1):
        lines.append(f"| {idx} | `{item['checkpoint']}` | {item['avg_score']:.4f} |")

    for idx, item in enumerate(top_results, start=1):
        lines.extend(
            [
                "",
                f"## Rank {idx}: `{item['checkpoint']}`",
                "",
                f"Average score: `{item['avg_score']:.4f}`",
                "",
                "| Scenario | Seed | Baseline Avg Wait | RL Avg Wait | Baseline Max Wait | RL Max Wait | Score |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for route_result in item["per_route"]:
            baseline = route_result["baseline"]
            rl = route_result["rl"]
            lines.append(
                "| "
                f"{route_result['scenario']} | "
                f"{route_result['seed']} | "
                f"{baseline['avg_waiting_time']:.2f} | "
                f"{rl['avg_waiting_time']:.2f} | "
                f"{baseline['max_waiting_time']:.2f} | "
                f"{rl['max_waiting_time']:.2f} | "
                f"{route_result['score']:.4f} |"
            )

        lines.extend(
            [
                "",
                "| Metric | Baseline Avg | RL Avg |",
                "|---|---:|---:|",
            ]
        )

        metric_keys = [
            ("arrived_vehicles", "Vehicles Arrived"),
            ("avg_queue_per_step", "Avg Queue / Step"),
            ("avg_lane_wait_per_step", "Avg Lane Wait / Step"),
            ("avg_trip_duration", "Avg Trip Duration"),
            ("avg_waiting_time", "Avg Waiting Time"),
            ("avg_time_loss", "Avg Time Loss"),
            ("max_waiting_time", "Max Waiting Time"),
        ]

        for key, label in metric_keys:
            baseline_avg = sum(r["baseline"][key] for r in item["per_route"]) / len(
                item["per_route"]
            )
            rl_avg = sum(r["rl"][key] for r in item["per_route"]) / len(
                item["per_route"]
            )
            lines.append(
                f"| {label} | {format_metric(baseline_avg)} | {format_metric(rl_avg)} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train for 200 episodes with periodic checkpoints and select the best "
            "checkpoint by average evaluation across multiple traffic scenarios."
        )
    )
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save one checkpoint every N episodes.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/final_training_v2",
        help="Directory used for training checkpoints and evaluation artifacts.",
    )
    parser.add_argument(
        "--model-path",
        default="dqn_model_final_v2.pth",
        help="Path to the final model saved after training.",
    )
    parser.add_argument(
        "--eval-scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=sorted(SCENARIOS),
        help="Traffic scenarios used during checkpoint evaluation.",
    )
    parser.add_argument(
        "--eval-seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds used to generate evaluation route files.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=7200,
        help="Simulation length used for generated evaluation traffic files.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Train again even if matching checkpoints already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    eval_route_dir = checkpoint_dir / "eval_routes"
    summary_path = checkpoint_dir / "checkpoint_summary.json"
    markdown_summary_path = checkpoint_dir / "checkpoint_summary.md"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    expected_checkpoint_count = args.episodes // args.checkpoint_every
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_ep*.pth"),
        key=checkpoint_sort_key,
    )
    final_model_path = Path(args.model_path)

    should_train = args.force_retrain
    if not should_train:
        if (
            len(checkpoints) < expected_checkpoint_count
            or not final_model_path.exists()
        ):
            should_train = True

    if should_train:
        train(
            episodes=args.episodes,
            model_path=args.model_path,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=str(checkpoint_dir),
        )
    else:
        print("Reusing existing checkpoints and final model. Skipping training.")

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_ep*.pth"), key=checkpoint_sort_key
    )
    if not checkpoints:
        raise RuntimeError("No checkpoints were created during training.")

    eval_routes = build_eval_routes(
        eval_route_dir,
        scenarios=args.eval_scenarios,
        seeds=args.eval_seeds,
        steps=args.eval_steps,
    )
    ranked_results = evaluate_checkpoints(checkpoints, eval_routes)
    best = ranked_results[0]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(ranked_results, f, indent=2)
    write_markdown_summary(ranked_results, markdown_summary_path, top_k=5)

    print(f"Best checkpoint: {best['checkpoint']}")
    print(f"Average score: {best['avg_score']:.4f}")
    print()
    print("Top 5 checkpoints:")
    for item in ranked_results[:5]:
        print(f"  {item['checkpoint']} | avg_score={item['avg_score']:.4f}")
    print()
    print(f"Detailed summary written to {summary_path}")
    print(f"Readable summary written to {markdown_summary_path}")


if __name__ == "__main__":
    main()

"""Score every checkpoint in a directory and pick the best one.

Usage:
    python scripts/select_best_checkpoint.py \\
        --checkpoint-dir runs/current/checkpoints \\
        --output runs/current/best_checkpoint.txt

Runs batch_evaluate.py against each checkpoint, computes a composite score that
balances arrivals (more is better) against waiting time / trip duration / queue
(less is better), and writes the winning checkpoint path to ``--output``. Each
per-checkpoint summary JSON is kept next to the checkpoint so the full evaluation
history is auditable.

Composite score formula (higher = better):

    score = arrived_vehicles
            - 0.5 * avg_waiting_time
            - 0.3 * avg_trip_duration
            - 5.0 * avg_queue_per_step

The weights were chosen so that on the current baseline run a +10% improvement
in arrivals is roughly equivalent to a -20% improvement in waiting time. Tune
the weights via ``--score-weights`` if your project values one metric more.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def composite_score(rl_avg, weights):
    return (
        weights["arrived"] * rl_avg["arrived_vehicles"]
        - weights["wait"] * rl_avg["avg_waiting_time"]
        - weights["trip"] * rl_avg["avg_trip_duration"]
        - weights["queue"] * rl_avg["avg_queue_per_step"]
    )


def per_scenario_summary(results):
    """Aggregate (arrivals, wait) per scenario for both RL and baseline."""
    by_scenario = {}
    for r in results:
        by_scenario.setdefault(r["scenario"], []).append(r)
    summary = {}
    for sc, items in by_scenario.items():
        n = len(items)
        summary[sc] = {
            "rl_arrived": sum(i["rl"]["arrived_vehicles"] for i in items) / n,
            "rl_wait": sum(i["rl"]["avg_waiting_time"] for i in items) / n,
            "baseline_arrived": sum(i["baseline"]["arrived_vehicles"] for i in items)
            / n,
            "baseline_wait": sum(i["baseline"]["avg_waiting_time"] for i in items) / n,
        }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score every checkpoint in a directory and write the best path."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing checkpoint_*.pth files.",
    )
    parser.add_argument(
        "--output",
        default="best_checkpoint.txt",
        help="Text file that will receive the winning checkpoint path.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3600,
        help="Seconds of demand to generate per scenario.",
    )
    parser.add_argument(
        "--end-time",
        type=int,
        default=7200,
        help="Simulated seconds per evaluation run.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Subset of scenarios to evaluate (default: all).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[41, 42, 43],
        help="Seeds to evaluate per scenario.",
    )
    parser.add_argument(
        "--route-dir",
        default="runs/batch_eval_routes",
        help="Directory used for shared evaluation route files.",
    )
    parser.add_argument(
        "--score-weights",
        default="1.0,0.5,0.3,5.0",
        help="Comma-separated arrived,wait,trip,queue weights.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing eval_*.json files instead of re-running.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    weights_list = [float(x) for x in args.score_weights.split(",")]
    if len(weights_list) != 4:
        raise SystemExit("--score-weights must have 4 comma-separated floats")
    weights = {
        "arrived": weights_list[0],
        "wait": weights_list[1],
        "trip": weights_list[2],
        "queue": weights_list[3],
    }

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise SystemExit(f"Checkpoint dir not found: {ckpt_dir}")
    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pth"))
    if not checkpoints:
        raise SystemExit(f"No checkpoint_*.pth files in {ckpt_dir}")

    results = []
    for ckpt in checkpoints:
        summary_path = ckpt.parent / f"eval_{ckpt.stem}.json"
        if not (args.skip_existing and summary_path.exists()):
            cmd = [
                sys.executable,
                "batch_evaluate.py",
                "--model-path",
                str(ckpt),
                "--steps",
                str(args.steps),
                "--end-time",
                str(args.end_time),
                "--route-dir",
                args.route_dir,
                "--seeds",
                *(str(s) for s in args.seeds),
                "--summary-json",
                str(summary_path),
            ]
            if args.scenarios:
                cmd += ["--scenarios", *args.scenarios]
            print(f"--- Evaluating {ckpt.name} ---")
            subprocess.run(cmd, check=True)

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        score = composite_score(data["rl_average"], weights)
        scen = per_scenario_summary(data["results"])
        results.append(
            {
                "checkpoint": str(ckpt),
                "score": score,
                "rl_average": data["rl_average"],
                "per_scenario": scen,
            }
        )
        print(f"  {ckpt.name}: score={score:.1f}")

    print()
    print("Checkpoint scoreboard (sorted by composite):")
    headers = ("Checkpoint", "Score", "Arrived", "AvgWait", "TripDur", "AvgQ")
    rows = []
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        m = r["rl_average"]
        rows.append(
            (
                Path(r["checkpoint"]).name,
                f"{r['score']:.1f}",
                f"{m['arrived_vehicles']:.0f}",
                f"{m['avg_waiting_time']:.1f}",
                f"{m['avg_trip_duration']:.1f}",
                f"{m['avg_queue_per_step']:.2f}",
            )
        )
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    print(
        "  ".join(
            f"{headers[i]:<{widths[i]}}" if i == 0 else f"{headers[i]:>{widths[i]}}"
            for i in range(len(headers))
        )
    )
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(
            "  ".join(
                f"{row[i]:<{widths[i]}}" if i == 0 else f"{row[i]:>{widths[i]}}"
                for i in range(len(row))
            )
        )

    best = max(results, key=lambda x: x["score"])
    print()
    print(f"Best checkpoint: {best['checkpoint']}  (score {best['score']:.1f})")
    print()
    print("Per-scenario breakdown for the winner (RL vs baseline arrivals/wait):")
    for sc, m in sorted(best["per_scenario"].items()):
        delta_arr = m["rl_arrived"] - m["baseline_arrived"]
        delta_wait = m["rl_wait"] - m["baseline_wait"]
        print(
            f"  {sc:18s}  "
            f"arrived RL={m['rl_arrived']:6.0f} (Δ {delta_arr:+5.0f})  "
            f"wait RL={m['rl_wait']:7.1f} (Δ {delta_wait:+7.1f})"
        )

    Path(args.output).write_text(best["checkpoint"], encoding="utf-8")
    print(f"\nWrote winning checkpoint path to {args.output}")


if __name__ == "__main__":
    main()

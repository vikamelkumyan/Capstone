"""One-command reproduction entrypoint for the capstone submission.

By default this script regenerates the code-owned result figures from the
committed final training log and evaluation summary. Use --run-evaluation to
rerun SUMO simulations before regenerating plots.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FINAL_ASSETS = ROOT / "code" / "visualization" / "final_results"
DEFAULT_MODEL = ROOT / "code" / "models" / "komitas_multi_intersection_dqn_ep075.pth"
DEFAULT_SUMMARY = FINAL_ASSETS / "eval_ep075_summary.json"
DEFAULT_TRAINING_LOG = FINAL_ASSETS / "training_log_v3.csv"


def run(cmd):
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def require_file(path, description):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def check_environment(require_sumo):
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required.")

    if require_sumo:
        missing = [
            binary for binary in ("sumo", "duarouter") if not shutil.which(binary)
        ]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"Missing SUMO command(s) on PATH: {joined}. "
                "Install Eclipse SUMO and set SUMO_HOME before rerunning."
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate final capstone figures and, optionally, rerun the "
            "full SUMO evaluation used for the reported results."
        )
    )
    parser.add_argument(
        "--run-evaluation",
        action="store_true",
        help=(
            "Rerun batch_evaluate.py across the reported scenario/seed grid. "
            "This can take a long time and requires SUMO."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL),
        help="Model checkpoint used when --run-evaluation is enabled.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(DEFAULT_SUMMARY),
        help="Evaluation summary JSON used by scripts/plot_evaluation.py.",
    )
    parser.add_argument(
        "--training-log",
        default=str(DEFAULT_TRAINING_LOG),
        help="Training CSV used by scripts/plot_training.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FINAL_ASSETS),
        help="Directory where reproduced figures are written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary_json = Path(args.summary_json)
    training_log = Path(args.training_log)
    model_path = Path(args.model_path)

    check_environment(require_sumo=args.run_evaluation)
    require_file(training_log, "final training log")
    require_file(model_path, "trained DQN model")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_evaluation:
        run(
            [
                sys.executable,
                "code/batch_evaluate.py",
                "--model-path",
                str(model_path),
                "--scenarios",
                "morning_rush",
                "evening_rush",
                "off_peak",
                "corridor_stress",
                "--seeds",
                "41",
                "42",
                "43",
                "--steps",
                "3600",
                "--end-time",
                "7200",
                "--route-dir",
                "runs/batch_eval_routes",
                "--summary-json",
                str(summary_json),
            ]
        )

    require_file(summary_json, "final evaluation summary")

    run(
        [
            sys.executable,
            "code/scripts/plot_training.py",
            "--log-csv",
            str(training_log),
            "--output",
            str(output_dir / "training_overview.png"),
            "--output-dir",
            str(output_dir),
        ]
    )
    run(
        [
            sys.executable,
            "code/scripts/plot_evaluation.py",
            "--summary-json",
            str(summary_json),
            "--output-dir",
            str(output_dir),
        ]
    )

    print()
    print("Reproduction complete.")
    print(f"Figures written to: {output_dir}")


if __name__ == "__main__":
    main()

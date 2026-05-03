"""Generate training curve plots from a CSV log produced by train.py --log-csv."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def moving_average(values, window=10):
    smoothed = []
    for i, v in enumerate(values):
        start = max(0, i - window + 1)
        chunk = [x for x in values[start : i + 1] if not math.isnan(x)]
        smoothed.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return smoothed


def scenario_name(label):
    """Extract scenario name from labels like 'morning_rush seed 42'."""
    return label.split(" seed ", 1)[0].strip()


def read_log(log_path):
    rows = []
    with log_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loss_raw = row.get("avg_loss", "nan")
            try:
                loss = float(loss_raw)
            except ValueError:
                loss = float("nan")
            rows.append(
                {
                    "episode": int(row["episode"]),
                    "total_reward": float(row["total_reward"]),
                    "avg_reward": float(
                        row.get("avg_reward_per_agent_decision", row["total_reward"])
                    ),
                    "avg_loss": loss,
                    "epsilon": float(row.get("epsilon", "nan")),
                    "scenario": scenario_name(row.get("scenario", "unknown")),
                }
            )
    return rows


def save_overview_plot(rows, output_path, window, plt):
    episodes = [r["episode"] for r in rows]
    avg_rewards = [r["avg_reward"] for r in rows]
    losses = [r["avg_loss"] for r in rows]
    epsilons = [r["epsilon"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("Komitas Corridor DQN Training Overview", fontsize=13)

    axes[0].plot(
        episodes, avg_rewards, alpha=0.25, color="tab:blue", linewidth=0.8, label="Raw"
    )
    axes[0].plot(
        episodes,
        moving_average(avg_rewards, window),
        color="tab:blue",
        linewidth=1.8,
        label=f"{window}-episode moving average",
    )
    axes[0].set_ylabel("Avg Reward / Agent Decision")
    axes[0].set_title("Training Reward")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    valid_loss = [
        (ep, loss) for ep, loss in zip(episodes, losses) if not math.isnan(loss)
    ]
    if valid_loss:
        loss_eps, loss_vals = zip(*valid_loss)
        axes[1].plot(
            loss_eps,
            loss_vals,
            alpha=0.25,
            color="tab:orange",
            linewidth=0.8,
            label="Raw",
        )
        axes[1].plot(
            loss_eps,
            moving_average(list(loss_vals), window),
            color="tab:orange",
            linewidth=1.8,
            label=f"{window}-episode moving average",
        )
    axes[1].set_ylabel("Average Loss (MSE)")
    axes[1].set_title("DQN Training Loss")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(episodes, epsilons, color="tab:green", linewidth=1.6)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].set_title("Exploration Schedule")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_scenario_reward_plot(rows, output_path, window, plt):
    by_scenario = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)

    fig, ax = plt.subplots(figsize=(11, 6))
    for scenario in sorted(by_scenario):
        scenario_rows = by_scenario[scenario]
        episodes = [r["episode"] for r in scenario_rows]
        rewards = [r["avg_reward"] for r in scenario_rows]
        ax.plot(episodes, rewards, alpha=0.22, linewidth=0.8)
        ax.plot(
            episodes,
            moving_average(rewards, min(window, max(len(rewards), 1))),
            linewidth=1.9,
            marker="o",
            markersize=2.5,
            label=scenario,
        )

    ax.set_title("Average Reward by Scenario")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward / Agent Decision")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_scenario_loss_plot(rows, output_path, window, plt):
    by_scenario = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)

    fig, ax = plt.subplots(figsize=(11, 6))
    for scenario in sorted(by_scenario):
        scenario_rows = [
            r for r in by_scenario[scenario] if not math.isnan(r["avg_loss"])
        ]
        if not scenario_rows:
            continue
        episodes = [r["episode"] for r in scenario_rows]
        losses = [r["avg_loss"] for r in scenario_rows]
        ax.plot(episodes, losses, alpha=0.22, linewidth=0.8)
        ax.plot(
            episodes,
            moving_average(losses, min(window, max(len(losses), 1))),
            linewidth=1.9,
            marker="o",
            markersize=2.5,
            label=scenario,
        )

    ax.set_title("Average Loss by Scenario")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Loss (MSE)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot training reward and loss curves from a training_log.csv file."
    )
    parser.add_argument(
        "--log-csv",
        default="training_log.csv",
        help="Path to the CSV file produced by train.py --log-csv.",
    )
    parser.add_argument(
        "--output",
        default="training_curve.png",
        help="Output image path for the overview plot.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for overview plus scenario-specific plots.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Moving average window size for smoothing.",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        return

    log_path = Path(args.log_csv)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        print("Run training with: python train.py --log-csv training_log.csv")
        return

    rows = read_log(log_path)
    if not rows:
        print(f"No rows found in {log_path}")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_overview_plot(rows, out_path, args.window, plt)
    print(f"Saved training curve to {out_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        overview_path = output_dir / "training_overview.png"
        rewards_path = output_dir / "reward_by_scenario.png"
        loss_path = output_dir / "loss_by_scenario.png"
        save_overview_plot(rows, overview_path, args.window, plt)
        save_scenario_reward_plot(rows, rewards_path, args.window, plt)
        save_scenario_loss_plot(rows, loss_path, args.window, plt)
        print(f"Saved overview plot to {overview_path}")
        print(f"Saved scenario reward plot to {rewards_path}")
        print(f"Saved scenario loss plot to {loss_path}")


if __name__ == "__main__":
    main()

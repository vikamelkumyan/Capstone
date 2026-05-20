# Reinforcement Learning for Adaptive Traffic Signal Control on the Komitas Avenue

A Double DQN agent that controls six traffic lights along the Komitas Avenue corridor in Yerevan using [SUMO](https://eclipse.dev/sumo/) and TraCI. Each intersection runs its own local policy; coordination is implicit through neighbor corridor features in the state vector. Against a fixed-time baseline across Rush Hour, Off-Peak, and Corridor Stress demand profiles, the trained agent reduces average waiting time by 5.9% and worst-case waiting time by 18%.

## Reproduce Results

No SUMO installation required to regenerate the reported figures:

```bash
python code/scripts/reproduce.py
```

To rerun the full SUMO simulations before plotting (requires Eclipse SUMO):

```bash
python code/scripts/reproduce.py --run-evaluation
```

## Repository Layout

```text
├── code/
│   ├── train.py                 # training entrypoint
│   ├── train_final.py           # checkpoint training + selection
│   ├── evaluate.py              # fixed-time vs MaxPressure vs RL
│   ├── evaluate_batch.py        # multi-scenario batch evaluation
│   ├── evaluate_single.py       # legacy single-intersection evaluation
│   ├── visualize.py             # SUMO GUI inspection
│   ├── models/
│   │   ├── dqn_multi_ep075.pth  # selected final multi-intersection model
│   │   └── dqn_single.pth       # legacy single-intersection model
│   ├── scripts/
│   │   ├── reproduce.py         # one-command reproduction
│   │   ├── generate_traffic.py  # demand generation via duarouter
│   │   ├── plot_training.py
│   │   ├── plot_evaluation.py
│   │   └── plot_single.py
│   └── visualization/final_results/
│       ├── training_log.csv
│       ├── evaluation_summary.json
│       └── evaluation_summary_single.json
├── data/
│   ├── raw_data/sumo_data/      # SUMO network, config, routes
│   └── processed_data/
├── paper/                       # LaTeX source, references, PDF
└── requirements.txt
```

## Setup

Clone the repository:

```bash
git clone https://github.com/vikamelkumyan/Capstone.git
cd Capstone
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Python 3.10+** is required.

### SUMO (optional — only needed for training and simulation reruns)

[Eclipse SUMO](https://eclipse.dev/sumo/) (Simulation of Urban MObility) is the traffic simulator this project runs inside. It models the road network, vehicles, and traffic light phases. The Python scripts communicate with it in real time via the TraCI API to observe queue states and apply signal changes during training and evaluation.

**SUMO is not needed to reproduce the reported figures** — those are regenerated from committed JSON/CSV files. It is only required if you want to retrain the model or rerun the SUMO simulations from scratch.

Installation:

| Platform | Command |
|---|---|
| macOS | `brew install sumo` |
| Ubuntu/Debian | `sudo apt install sumo sumo-tools` |
| Windows | Download the installer from [sumo.dlr.de/docs/Downloads.php](https://sumo.dlr.de/docs/Downloads.php) |

After installation, set the `SUMO_HOME` environment variable to the SUMO installation directory (the installer does this automatically on Windows; on macOS/Linux add it to your shell profile):

```bash
export SUMO_HOME="/usr/share/sumo"   # adjust path to your installation
```

Full installation guide: [sumo.dlr.de/docs/Installing](https://sumo.dlr.de/docs/Installing/index.html)

## Workflow

All commands assume the venv is active and are run from the repo root.

### 1. Generate demand

```bash
python code/scripts/generate_traffic.py \
  --scenario off_peak --seed 42 --steps 3600 \
  --output data/raw_data/sumo_data/routes.rou.xml \
  --net-file data/raw_data/sumo_data/komitas.net.xml
```

Available scenarios: `off_peak`, `rush_hour`, `corridor_stress`. Use `--demand-scale` to amplify volume.

### 2. Train

```bash
python code/train.py
```

Trains headless, round-robin across scenarios and seeds, saves periodic checkpoints. Key options:

```bash
python code/train.py \
  --episodes 200 \
  --decisions-per-episode 720 \
  --checkpoint-every 25 \
  --checkpoint-dir runs/checkpoints \
  --log-csv runs/training_log.csv
```

Smoke test: `python code/train.py --episodes 1 --decisions-per-episode 1`

### 3. Evaluate

```bash
python code/evaluate.py        # fixed-time vs MaxPressure vs RL, single scenario
python code/evaluate_batch.py  # full scenario/seed grid with summary table
```

Batch evaluation with explicit model and output:

```bash
python code/evaluate_batch.py \
  --model-path code/models/dqn_multi_ep075.pth \
  --scenarios rush_hour off_peak corridor_stress \
  --seeds 41 42 43 \
  --summary-json runs/evaluation_summary.json
```

### 4. Checkpoint selection

```bash
python code/train_final.py
```

Trains with periodic checkpoints, evaluates each across scenarios, and ranks by weighted score.

### 5. GUI inspection

```bash
python code/visualize.py
python code/visualize.py --route-file data/raw_data/sumo_data/routes.rou.xml --decisions 100 --render-delay 0.25
```

### 6. Legacy single-intersection

```bash
python code/evaluate_single.py
python code/evaluate_single.py \
  --json-output code/visualization/final_results/evaluation_summary_single.json
```

Uses the preserved Komitas-Vagharshyan single-intersection SUMO files under `data/raw_data/sumo_data/legacy_single_intersection/`.

## Controller Design

One local Double DQN per intersection (128 → 128 → action_dim MLP). Six intersections run simultaneously: `Komitas-Gyulbenkyan`, `Komitas-Vagharshyan`, `Komitas-Papazyan`, `Komitas-Vracakan`, `Komitas-Griboyedov`, `Komitas-Tigranyan`.

**State** (`k + 13` dims): per-phase queue pressures, elapsed green ratio, phase cycle position, local queue / wait / spillback, network load, and three features each for the upstream and downstream neighbor.

**Actions:** extend current green or switch to any valid green phase. Switches before `MIN_GREEN = 10 s` are masked.

**Reward:** PressLight-style pressure terms plus a starvation guard (ramps sharply when any lane exceeds 20 s wait) and a demand-scaled switch penalty (suppresses thrashing under low demand). A per-episode demand normalizer equalizes gradient magnitudes across scenarios.

**Training:** 400 episodes, round-robin scenario schedule (seeds 41–43), replay buffer 100k, Adam lr = 3e-4, gradient clip norm = 10, ε decays over the first 85% of episodes.

## Results

### Single Intersection — Komitas-Vagharshyan (Rush Hour)

Model: `dqn_single.pth`, evaluated on the preserved single-intersection setup.

| Metric | Fixed-Time | RL | Improvement |
|---|---:|---:|---:|
| Avg Queue / Step | 3.14 | 2.54 | +19.1% |
| Avg Trip Duration | 17.15 s | 15.55 s | +9.3% |
| Avg Waiting Time | 8.70 s | 7.04 s | +19.1% |
| Avg Time Loss | 12.27 s | 10.67 s | +13.0% |
| Max Waiting Time | 139.0 s | 48.0 s | +65.5% |

The 65.5% reduction in worst-case waiting time confirms the agent eliminated the starvation of low-demand approaches that plagues fixed-time schedules.

### Multi-Intersection — Komitas Corridor (averaged across scenarios and seeds)

Model: `dqn_multi_ep075.pth` — episode 75 of 500, chosen by multi-scenario weighted evaluation score.

| Metric | Fixed-Time | MaxPressure | RL | Improvement |
|---|---:|---:|---:|---:|
| Avg Queue / Step | 16.02 | 13.80 | 14.55 | +9.2% |
| Avg Trip Duration | 908.96 s | 890.21 s | 858.76 s | +5.5% |
| Avg Waiting Time | 722.44 s | 708.48 s | 679.79 s | +5.9% |
| Avg Time Loss | 811.30 s | 790.42 s | 761.13 s | +6.2% |
| Max Waiting Time | 4288.50 s | 5445.83 s | 3515.83 s | +18.0% |

RL outperforms both baselines on Rush Hour and Off-Peak demand.

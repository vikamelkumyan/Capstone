# Multi-Intersection Traffic Signal Control with DQN (SUMO + TraCI)

This repository trains and evaluates a Deep Q-Network (DQN) traffic signal controller for a multi-intersection Komitas corridor in SUMO.

The current active setup controls six named traffic lights:

- `Komitas-Gyulbenkyan`
- `Komitas-Vagharshyan`
- `Komitas-Papazyan`
- `Komitas-Vracakan`
- `Komitas-Griboyedov`
- `Komitas-Tigranyan`

Each controlled intersection has its own local DQN policy, and all six run together in the same simulation.

## Repository Layout

- [train.py](train.py:1): multi-intersection training entrypoint
- [evaluate.py](evaluate.py:1): fixed-time vs MaxPressure vs RL comparison
- [test_sim.py](test_sim.py:1): GUI visualization of the trained controller
- [batch_evaluate.py](batch_evaluate.py:1): evaluation across multiple scenarios and seeds
- [run_final_training.py](run_final_training.py:1): checkpoint-based final training and selection
- [scripts/generate_traffic.py](scripts/generate_traffic.py:1): demand generation using trips plus `duarouter`
- [scripts/inspect_network.py](scripts/inspect_network.py:1): SUMO traffic-light inspection utility
- [models/komitas_dqn_ep075.pth](models/komitas_dqn_ep075.pth): selected final DQN model bundle
- [sumo_data/komitas.net.xml](sumo_data/komitas.net.xml:1): active Komitas network
- [sumo_data/komitas.sumocfg](sumo_data/komitas.sumocfg:1): active SUMO config
- [sumo_data/routes.rou.xml](sumo_data/routes.rou.xml:1): active route demand

Legacy single-intersection Vagharshyan assets are preserved under `sumo_data/legacy_single_intersection/` for project history and comparison. The active code defaults to the Komitas multi-intersection setup.

## Requirements

- Python 3.10+
- Eclipse SUMO installed with `sumo`, `sumo-gui`, and `duarouter` on `PATH`
- `SUMO_HOME` configured
- Python packages from [requirements.txt](requirements.txt:1)

On macOS:

```bash
brew install sumo
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional:

```bash
pre-commit install
```

## Main Workflow

Run commands from the repository root.

### 1. Generate or refresh demand

```bash
./.venv/bin/python scripts/generate_traffic.py \
  --scenario morning_rush \
  --seed 42 \
  --steps 3600 \
  --output sumo_data/routes.rou.xml \
  --net-file sumo_data/komitas.net.xml
```

The script writes trips first, then uses `duarouter` to compute valid full routes on the Komitas network.

For a heavier congestion test, use the built-in corridor stress profile or scale demand upward:

```bash
./.venv/bin/python scripts/generate_traffic.py \
  --scenario corridor_stress \
  --demand-scale 1.2 \
  --seed 42 \
  --steps 3600 \
  --output sumo_data/routes.rou.xml \
  --net-file sumo_data/komitas.net.xml
```

### 2. Train

```bash
./.venv/bin/python train.py
```

Training runs headless with `sumo` by default. It generates multiple traffic scenarios, trains on a balanced scenario/seed schedule, saves periodic checkpoints, and saves the final model bundle to `dqn_model.pth` unless another `--model-path` is provided.

Example training run used for the current best result:

```bash
./.venv/bin/python train.py \
  --episodes 200 \
  --decisions-per-episode 720 \
  --checkpoint-every 25 \
  --checkpoint-dir runs/v3/checkpoints \
  --model-path runs/v3/dqn_model.pth \
  --training-route-dir runs/v3/training_routes \
  --log-csv runs/v3/training_log.csv
```

Quick smoke test:

```bash
./.venv/bin/python train.py --episodes 1 --decisions-per-episode 1
```

### 3. Visual test

```bash
./.venv/bin/python test_sim.py
```

This opens `sumo-gui` and prints the per-intersection local states, Q-values, and chosen actions.

Useful options:

```bash
./.venv/bin/python test_sim.py --route-file sumo_data/routes.rou.xml --decisions 100 --render-delay 0.25
```

### 4. Baseline vs RL comparison

```bash
./.venv/bin/python evaluate.py
```

This runs:

- fixed-time SUMO controller on the full Komitas corridor
- MaxPressure adaptive controller on the same scenario
- RL controller on the same scenario

and prints a side-by-side metric table.

### 5. Multi-scenario batch evaluation

```bash
./.venv/bin/python batch_evaluate.py
```

This generates a scenario/seed grid, runs fixed-time, MaxPressure, and RL on every route file, and prints:

- per-route results
- per-scenario controller comparison
- averaged fixed-time vs MaxPressure vs RL results
- RL policy diagnostics, including action dominance and switch counts

Example:

```bash
./.venv/bin/python batch_evaluate.py \
  --model-path models/komitas_dqn_ep075.pth \
  --scenarios morning_rush evening_rush off_peak corridor_stress \
  --seeds 41 42 43 \
  --steps 3600 \
  --end-time 7200 \
  --summary-json runs/v3/eval_ep075_summary.json
```

### 6. Checkpoint-based final training

```bash
./.venv/bin/python run_final_training.py
```

This workflow:

- trains for many episodes
- saves periodic checkpoints
- evaluates each checkpoint across multiple generated scenarios
- ranks checkpoints by averaged score

## Current Controller Design

The current controller:

- uses one local **Double DQN** per controlled intersection
- observes a `k+13`-dimensional local state per intersection: per-phase pressures, timing context, spillback pressure, network load, and neighbor corridor features
- can extend the current green or switch to a valid green phase directly
- masks invalid actions so early switches are not selected before `MIN_GREEN`
- synthesizes pair-specific yellow transitions from the current and target green states
- enforces `MIN_GREEN` and `MAX_GREEN`
- uses a pressure/wait/spillback reward with starvation protection and demand-scaled switching cost
- trains on a balanced round-robin scenario schedule
- uses gradient clipping (`max_norm=10.0`) for training stability
- epsilon decays over the first 85% of training episodes

This is a hybrid design: RL chooses local actions, while timing safety rules remain rule-based. The neighbor state features provide implicit corridor coordination without requiring a centralized joint policy.

## Current Evaluation Result

The selected final model is [models/komitas_dqn_ep075.pth](models/komitas_dqn_ep075.pth), selected from multi-scenario evaluation across four demand profiles and three seeds (`41`, `42`, `43`). Against the fixed-time baseline, this checkpoint improves average queue, trip duration, waiting time, time loss, and worst-case waiting time, while still arriving fewer vehicles overall.

| Metric | Fixed-Time | MaxPressure | RL | Delta vs Fixed |
|---|---:|---:|---:|---:|
| Vehicles Arrived | 2664.42 | 2313.75 | 2492.00 | -6.5% |
| Avg Queue / Step | 16.02 | 13.80 | 14.55 | +9.2% |
| Avg Trip Duration | 908.96 | 890.21 | 858.76 | +5.5% |
| Avg Waiting Time | 722.44 | 708.48 | 679.79 | +5.9% |
| Avg Time Loss | 811.30 | 790.42 | 761.13 | +6.2% |
| Max Waiting Time | 4288.50 | 5445.83 | 3515.83 | +18.0% |

Scenario-level behavior is mixed: RL performs strongly on `evening_rush` and `off_peak`, is competitive on `corridor_stress`, and remains weaker than fixed-time on `morning_rush`. The throughput gap and morning-rush degradation are the main current limitations.

## Active Config And Assets

Current defaults:

- config: [sumo_data/komitas.sumocfg](sumo_data/komitas.sumocfg:1)
- network: [sumo_data/komitas.net.xml](sumo_data/komitas.net.xml:1)
- route file: [sumo_data/routes.rou.xml](sumo_data/routes.rou.xml:1)
- legacy single-intersection files: [sumo_data/legacy_single_intersection/](sumo_data/legacy_single_intersection/README.md:1)

## Notes

- `train.py` uses headless `sumo`; `test_sim.py` uses `sumo-gui`
- route generation now depends on `duarouter`
- generated artifacts such as checkpoints, training routes, and `*.rou.alt.xml` files are ignored by git
- final report assets are stored under `docs/assets/final_results/`

## More Documentation

- [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md:1): complete capstone paper and final report source
- [docs/RL_OVERVIEW.md](docs/RL_OVERVIEW.md:1)
- [docs/PIPELINE.md](docs/PIPELINE.md:1)

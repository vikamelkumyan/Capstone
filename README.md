# Traffic Signal Control with DQN (SUMO + TraCI)

This repository trains and evaluates a Deep Q-Network (DQN) traffic light controller for the Komitas-Vagharshyan intersection using Eclipse SUMO and TraCI.

The repository is organized so a reviewer can reproduce the milestone directly from the tracked SUMO network, tracked route file, and the Python scripts in the root.

## What is in the repository

- `train.py`: trains the DQN controller and saves the weights to `dqn_model.pth`
- `test_sim.py`: loads `dqn_model.pth` and runs the controller in SUMO GUI mode
- `sumo_data/`: SUMO network, routes, and simulation configuration
- `scripts/`: helper scripts for inspecting the traffic light setup and generated traffic
- `requirements.txt`: Python dependencies for a fresh environment

## Requirements

- Python 3.10+
- Eclipse SUMO installed with `sumo` and `sumo-gui` available on `PATH`
- `SUMO_HOME` set correctly
- Python packages: `torch`, `traci`, `numpy`

On macOS, SUMO can be installed with:

```bash
brew install sumo
```

## Setup

If you want to create a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Run commands from the repository root.

### 1. Train the model

```bash
./.venv/bin/python train.py
```

This starts SUMO, samples from multiple generated traffic scenarios during training, and writes the trained weights to `dqn_model.pth`.
Training runs headless by default, uses fixed seeds for reproducibility, and can generate per-episode route files under `training_routes/`.

### 2. Test the trained model

```bash
./.venv/bin/python test_sim.py
```

This opens `sumo-gui`, loads `dqn_model.pth`, and runs the trained controller while printing each decision to the terminal.
The current controller can either extend the active green or switch directly to one of the valid green phases, while still respecting minimum-green and synthesized yellow-transition logic.
You can also override the model, route file, decision count, and GUI speed:

```bash
./.venv/bin/python test_sim.py --route-file sumo_data/routes.rou.xml --decisions 150 --render-delay 0.25
```

### 3. Evaluate RL against the default SUMO controller

```bash
./.venv/bin/python evaluate.py
```

This runs the same scenario twice in headless SUMO:

- once with the default built-in traffic light program
- once with the trained RL controller

It then prints side-by-side metrics such as average queue, waiting time, trip duration, and time loss.

### 4. Final training with checkpoint selection

```bash
./.venv/bin/python run_final_training.py
```

This does the full milestone-selection loop:

- trains for `200` episodes
- saves checkpoints every `10` episodes
- trains on multiple generated traffic scenarios per episode
- generates multiple evaluation traffic scenarios
- evaluates every checkpoint with `evaluate.py` logic
- picks the checkpoint with the best average cross-scenario score

### 5. Batch evaluation across multiple scenarios

```bash
./.venv/bin/python batch_evaluate.py
```

This generates a grid of evaluation route files, runs baseline and RL on each one, and prints:

- a per-route summary
- one averaged comparison table across all scenarios and seeds

Example:

```bash
./.venv/bin/python batch_evaluate.py --scenarios morning_rush evening_rush off_peak --seeds 41 42 43 --summary-json checkpoints/batch_eval_summary.json
```

## Results

Results now depend on the selected training and evaluation artifacts. The recommended workflow is:

1. run `./.venv/bin/python run_final_training.py`
2. inspect the generated leaderboard in `checkpoints/final_training_v2/checkpoint_summary.json`
3. run `./.venv/bin/python batch_evaluate.py --model-path <best-checkpoint>`
4. use the averaged multi-scenario results for final comparison and reporting

This avoids hard-coding stale numbers in the README when the training setup changes.

## Controller Summary

The current RL controller:

- observes queue demand for the five valid green phases plus timing context
- can extend the current green or request a specific next green phase
- inserts a pair-specific yellow transition synthesized from the current and target green states
- still enforces `MIN_GREEN` and `MAX_GREEN`

This gives the agent more useful control authority than a simple fixed-cycle extend/switch controller while keeping signal changes constrained.

## Useful notes

- `train.py` uses headless `sumo` by default. `test_sim.py` uses `sumo-gui` for visualization.
- If you want to override the SUMO binary, set `SUMO_BINARY` before running:

```bash
SUMO_BINARY=sumo ./.venv/bin/python train.py
```

- `test_sim.py` requires `dqn_model.pth`. Train first if the file does not exist.
- The main SUMO config is `sumo_data/komitas-vagharshyan.sumocfg`.
- The tracked `sumo_data/komitas-vagharshyan.net.xml` and `sumo_data/routes.rou.xml` files are intentional submission assets, not temporary build output.

## Helper scripts

- `./.venv/bin/python scripts/inspect_tls.py`: print traffic light IDs and phase states
- `./.venv/bin/python scripts/check_tls.py`: verify the configured traffic light mapping
- `./.venv/bin/python scripts/analyze_phases.py`: inspect phase behavior
- `./.venv/bin/python scripts/generate_traffic.py --scenario morning_rush`: regenerate route demand data deterministically if needed

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

This starts SUMO, runs 100 training episodes, and writes the trained weights to `dqn_model.pth`.
Training uses a fixed random seed for reproducibility.

### 2. Test the trained model

```bash
./.venv/bin/python test_sim.py
```

This opens `sumo-gui`, loads `dqn_model.pth`, and runs the trained controller while printing each decision to the terminal.

### 3. Evaluate RL against the default SUMO controller

```bash
./.venv/bin/python evaluate.py
```

This runs the same scenario twice in headless SUMO:

- once with the default built-in traffic light program
- once with the trained RL controller

It then prints side-by-side metrics such as average queue, waiting time, trip duration, and time loss.

## Results

On the tracked scenario in this repository, the trained RL controller outperformed the default SUMO controller while keeping the same throughput.

| Metric | Baseline | RL |
|---|---:|---:|
| Steps | 12022 | 11729 |
| Vehicles Arrived | 2250 | 2250 |
| Avg Queue / Step | 1.56 | 1.53 |
| Avg Lane Wait / Step | 33.17 | 29.07 |
| Avg Trip Duration | 16.61 | 16.15 |
| Avg Waiting Time | 7.65 | 7.20 |
| Avg Time Loss | 11.32 | 10.86 |
| Max Waiting Time | 76.00 | 72.00 |

Summary:

- throughput stayed equal at `2250` arrived vehicles
- RL slightly reduced average queue length
- RL reduced average waiting time, trip duration, time loss, and worst-case waiting time

These numbers are from one tracked demand scenario. For a stronger evaluation, run the same comparison across multiple generated traffic scenarios and random seeds.

## Useful notes

- The scripts default to `sumo-gui`. If you want to use a different SUMO binary, set `SUMO_BINARY` before running:

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

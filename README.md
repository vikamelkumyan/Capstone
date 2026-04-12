# Traffic Signal Control with Q-Learning (SUMO + TraCI)

This repository contains an RL-based Traffic Light controller implementation for the Komitas-Vagharshyan intersection using Eclipse SUMO and tabular Q-Learning.

## Project Structure

- `train.py`: Main Q-learning script. Connects to SUMO via TraCI and trains the model.
- `test_sim.py`: Inference script to run and evaluate the trained model (`q_table.npy`).
- `sumo_data/`: Contains the SUMO network, demands, routes, and `.sumocfg` configurations.
- `scripts/`: Helper scripts used during development for analyzing TLS IDs and phase combinations.

## Setup Requirements

1. Install Eclipse SUMO:
    ```bash
    # MacOS
    brew install sumo
    ```
2. Make sure you have the SUMO_HOME environment variable set.
3. Install Python dependencies:
    ```bash
    pip install traci numpy
    ```

## How to Run

### Training
To train the model from scratch (this will output or overwrite `q_table.npy`):

```bash
python train.py
```

### Testing
To watch a simulation run with the output of the trained model (Ensure you have `q_table.npy` generated):

```bash
python test_sim.py
```

## Useful Tools

If you need to analyze the environment:
- `python scripts/inspect_tls.py`: Print all Traffic Light IDs and phase states.
- `python scripts/check_tls.py`: Ensure the correct mapping mapping is running.

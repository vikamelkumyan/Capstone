# Pipeline

This file explains the current end-to-end workflow from tracked SUMO assets to a trained RL controller and comparison results.

## 1. Scenario Assets

The tracked SUMO assets live under [sumo_data](/Users/macbook/Documents/komitas-vagharshyan/sumo_data):

- `komitas-vagharshyan.net.xml`: network, junction topology, and traffic light logic
- `komitas-vagharshyan.sumocfg`: SUMO scenario configuration
- `routes.rou.xml`: active route demand file
- `komitas-vagharshyan.osm`: raw map source

All main scripts run against:

```text
sumo_data/komitas-vagharshyan.sumocfg
```

## 2. Demand Generation

Traffic demand can be regenerated with [scripts/generate_traffic.py](/Users/macbook/Documents/komitas-vagharshyan/scripts/generate_traffic.py:1).

Example:

```bash
./.venv/bin/python scripts/generate_traffic.py --scenario evening_rush --seed 42 --steps 7200 --output sumo_data/routes.rou.xml
```

[train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:1) can also generate deterministic route banks under `training_routes/` when training across multiple scenarios.

## 3. SUMO Startup

When training or evaluation starts, the code:

1. prepares a runnable SUMO config, optionally overriding the route file
2. rewrites relative SUMO asset paths to absolute paths for temporary configs
3. starts SUMO through TraCI
4. identifies the controlled traffic light
5. builds a phase-to-lane mapping for the valid green phases

That mapping is important because RL decisions are phase-level, but the traffic measurements come from lanes.

## 4. State Extraction

At each decision point, the controller builds a 7-feature state:

- normalized demand for each of the 5 green phases
- normalized elapsed green time
- normalized current phase position

This happens in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:215).

## 5. Action Selection

During training:

- epsilon-greedy exploration is used
- early episodes explore more
- later episodes rely more on the DQN

During testing and evaluation:

- the highest-Q action is chosen greedily

The action space is:

- `EXTEND`
- switch directly to one of the valid green phases

This is decoded in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:287).

## 6. Transition Handling

Before switching from one green phase to another, the controller synthesizes a yellow transition from the current and target green state strings.

That transition:

- keeps movements green if they stay active in the target phase
- sets withdrawn movements to yellow
- keeps everything else red

Then the controller applies the target green state. This logic lives in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:251) and is reused by [evaluate.py](/Users/macbook/Documents/komitas-vagharshyan/evaluate.py:1).

## 7. Reward And Learning

After each decision, the code measures:

- total queue
- worst-lane queue
- newly arrived vehicles
- newly loaded vehicles

The reward then favors:

- queue reduction
- some protection against worst-lane starvation
- moving vehicles through the network

and penalizes:

- inflow pressure
- excessive green holding
- unnecessary switching

Transitions are stored in replay memory, minibatches are sampled, and the DQN is updated against a target network.

## 8. Outputs

Normal training writes:

```text
dqn_model.pth
```

The checkpoint-based final workflow can also write:

- periodic checkpoints under `checkpoints/`
- a final selected model such as `dqn_model_final_v2.pth`
- ranking summaries under the checkpoint directory

## 9. Visual Test

[test_sim.py](/Users/macbook/Documents/komitas-vagharshyan/test_sim.py:1) runs the trained controller in `sumo-gui`.

Its role is qualitative:

- watch the signal behavior
- inspect printed Q-values and actions
- verify the controller is switching sensibly

## 10. Quantitative Evaluation

[evaluate.py](/Users/macbook/Documents/komitas-vagharshyan/evaluate.py:1) compares:

- the default SUMO traffic light controller
- the trained RL controller

It reports:

- steps to clear the scenario
- vehicles arrived
- average queue per step
- average lane wait per step
- average trip duration
- average waiting time
- average time loss
- maximum waiting time

## 11. Final-Selection Workflow

[run_final_training.py](/Users/macbook/Documents/komitas-vagharshyan/run_final_training.py:1) automates the full selection loop:

1. train for a fixed number of episodes
2. save checkpoints periodically
3. generate evaluation route files
4. evaluate each checkpoint across those routes
5. rank checkpoints by average cross-scenario score
6. write JSON and Markdown summaries

This is the best workflow when you want a final submission model instead of just the last checkpoint.

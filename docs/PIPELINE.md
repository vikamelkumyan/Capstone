# Pipeline

This file explains the current multi-intersection Komitas workflow from SUMO assets to training and evaluation.

## 1. Active Scenario Files

The active defaults are:

- [sumo_data/komitas.net.xml](sumo_data/komitas.net.xml:1)
- [sumo_data/komitas.sumocfg](sumo_data/komitas.sumocfg:1)
- [sumo_data/routes.rou.xml](sumo_data/routes.rou.xml:1)

The route file is compatible with the Komitas network because it is generated as trips and then routed through the network using `duarouter`.

## 2. Demand Generation

[scripts/generate_traffic.py](scripts/generate_traffic.py:1) creates traffic in two stages:

1. generate trips from scenario probabilities
2. run `duarouter` to compute valid full routes on the chosen network

This avoids invalid direct edge-to-edge route definitions on larger corridor networks.

## 3. SUMO Startup

The training and evaluation code:

1. loads the active SUMO config
2. optionally overrides the route file
3. rewrites relative file inputs to absolute paths for temporary configs
4. starts SUMO through TraCI

The active default config is [sumo_data/komitas.sumocfg](sumo_data/komitas.sumocfg:1).

## 4. Controlled TLS Set

At startup the code builds the active controlled set from [train.py](train.py:17):

- `Komitas-Gyulbenkyan`
- `Komitas-Vagharshyan`
- `Komitas-Papazyan`
- `Komitas-Vracakan`
- `Komitas-Griboyedov`
- `Komitas-Tigranyan`

For each one, the code parses its traffic light logic and derives:

- valid stable green phases
- nominal next-green transitions
- local state and action dimensions

## 5. Local State Extraction

For each controlled TLS, the code:

1. maps green phases to their incoming lanes
2. measures halting vehicles on those lanes
3. builds a local state vector

This gives one local state per intersection, not one giant centralized state.

## 6. Local Action Selection

For each controlled TLS, a local DQN chooses:

- `EXTEND`
- or a direct target green phase

During training, actions are epsilon-greedy.
During testing and evaluation, actions are greedy.

## 7. Simultaneous Corridor Control

All local actions are applied together at each control step.

The control loop:

1. build local states for all controlled TLSs
2. choose local actions for all controlled TLSs
3. determine which TLSs need to switch
4. apply yellow transitions for all switching TLSs
5. apply target green states
6. advance the simulation by the extension interval

This means the corridor is coordinated at the simulation-step level even though the policies are local.

## 8. Reward And Learning

Each controlled TLS gets a local reward built from:

- local queue reduction
- local worst-lane improvement
- a shared global throughput term
- a small inflow penalty
- an excess-green penalty
- a switch penalty

Each TLS stores its own transitions in its own replay buffer, and each local DQN is trained from its own samples.

## 9. Model Output

The saved model is now a bundle, not one plain single-intersection state dict.

The bundle stores:

- controlled TLS IDs
- one local state dict per TLS
- local dimensions and metadata

This is why the training, testing, and evaluation code all use shared load/save helpers rather than raw `torch.load()` into a single network.

## 10. Visual Testing

[test_sim.py](test_sim.py:1) loads the model bundle and runs all six local controllers in `sumo-gui`.

It prints, per decision:

- TLS ID
- current phase
- elapsed green
- local state
- local Q-values
- chosen action

## 11. Quantitative Evaluation

[evaluate.py](evaluate.py:1) compares:

- default SUMO control over the corridor
- MaxPressure adaptive control over the corridor
- the trained RL control over the corridor

The reported metrics are aggregated over the whole corridor. RL evaluation also records action counts, switch counts, and blocked-switch counts for diagnosing policy collapse or excessive phase changing.

## 12. Multi-Scenario Evaluation

[batch_evaluate.py](batch_evaluate.py:1) runs the same fixed-time, MaxPressure, and RL comparison over multiple scenarios and seeds and then averages the results.

The current final-evaluation command is:

```bash
python batch_evaluate.py \
    --model-path models/komitas_dqn_ep075.pth \
    --scenarios morning_rush evening_rush off_peak corridor_stress \
    --seeds 41 42 43 \
    --steps 3600 \
    --end-time 7200 \
    --summary-json runs/v3/eval_ep075_summary.json
```

The batch output includes per-route results, per-scenario means, overall mean ± std, and RL policy diagnostics.

## 13. Final Selection Workflow

[run_final_training.py](run_final_training.py:1) automates:

1. training with periodic checkpoints
2. scenario generation
3. checkpoint evaluation
4. checkpoint ranking
5. summary export

This is the strongest current workflow for selecting a final corridor controller.

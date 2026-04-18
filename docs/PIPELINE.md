# Pipeline

This file explains the end-to-end pipeline from SUMO assets to a trained policy.

## 1. Static Scenario Assets

The tracked scenario files live in [sumo_data](/Users/macbook/Documents/komitas-vagharshyan/sumo_data):

- `komitas-vagharshyan.osm`: raw map source
- `komitas-vagharshyan.net.xml`: SUMO network with junction logic and traffic-light phases
- `routes.rou.xml`: active route demand used by training and testing
- `komitas-vagharshyan.sumocfg`: SUMO configuration file

The Python code runs the scenario described by:

```text
sumo_data/komitas-vagharshyan.sumocfg
```

## 2. Route Generation

If traffic demand needs to be regenerated, use [scripts/generate_traffic.py](/Users/macbook/Documents/komitas-vagharshyan/scripts/generate_traffic.py:1).

It builds a route file from:

- a named scenario such as `morning_rush`
- a deterministic random seed
- a fixed set of valid source-destination movements

Example:

```bash
./.venv/bin/python scripts/generate_traffic.py --scenario morning_rush --seed 42
```

The output is normally:

```text
sumo_data/routes.rou.xml
```

## 3. Environment Initialization

When training or testing begins, the code:

1. Starts SUMO through TraCI
2. Loads the configured network
3. Finds the target traffic-light system
4. Builds a mapping from green phases to incoming lanes

That mapping matters because the reward and state depend on lane-level queue information, but the agent chooses actions at the phase level.

## 4. State Extraction

At every decision point, the controller extracts:

- current phase demand
- next phase demand
- competing phase demand
- elapsed green time

This converts raw SUMO measurements into a compact RL state vector.

## 5. Action Selection

During training:

- the controller uses epsilon-greedy exploration
- random actions are used early
- model-based actions dominate later

During testing:

- the controller always picks the action with the highest Q-value

The action is then passed through the rule layer:

- no switch before `MIN_GREEN`
- forced switch at `MAX_GREEN`

## 6. Environment Transition

Once the action is resolved:

- the simulation is advanced by `EXTEND_STEP`
- or the controller moves through the built-in yellow phase and then into the next green phase

After the transition, new queue and waiting-time statistics are measured.

## 7. Reward And Learning

Training then computes the reward from congestion change and updates the replay buffer.

When enough samples exist:

1. a minibatch is sampled
2. target Q-values are computed using the target network
3. the policy network is optimized with MSE loss

This repeats for every decision in every episode.

## 8. Model Output

At the end of training, the learned policy network weights are written to:

```text
dqn_model.pth
```

That file is the only learned artifact needed for inference.

## 9. Evaluation Run

[test_sim.py](/Users/macbook/Documents/komitas-vagharshyan/test_sim.py:9) loads `dqn_model.pth` and runs the trained controller in SUMO.

Its purpose is qualitative and debugging-oriented:

- it shows the GUI
- it prints states and Q-values
- it reveals when rule constraints override the model

## 10. Practical Mental Model

A useful way to read the pipeline is:

- SUMO provides traffic dynamics
- helper logic converts them into a phase-level RL state
- the DQN proposes extend or switch
- rule-based timing constraints sanitize that proposal
- SUMO executes the result
- the congestion change becomes reward

So the final system is not purely learned and not purely rule-based. It is a hybrid controller with a learned decision layer inside a fixed operational envelope.

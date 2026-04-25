# RL Overview

This project uses a Deep Q-Network (DQN) to control one SUMO traffic light at the Komitas-Vagharshyan intersection. The controller is not fully unconstrained: it chooses when to keep the current green and which valid green phase to serve next, while timing safety rules still enforce minimum green and yellow clearance.

## Problem Setup

- Environment: Eclipse SUMO through TraCI
- Controlled junction: `cluster_11668441165_11668441166_11668441167_2912634528_#8more`
- Training entrypoint: [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:1)
- Inference entrypoint: [test_sim.py](/Users/macbook/Documents/komitas-vagharshyan/test_sim.py:1)
- Output model: `dqn_model.pth`

The current controller operates on these green phases from [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:19):

- phase `0`
- phase `2`
- phase `4`
- phase `5`
- phase `7`

## State Representation

The state is created in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:215) and has 7 continuous features:

1. normalized demand for green phase `0`
2. normalized demand for green phase `2`
3. normalized demand for green phase `4`
4. normalized demand for green phase `5`
5. normalized demand for green phase `7`
6. normalized elapsed green time
7. normalized position of the current green phase in the cycle

The state vector is:

```python
[d_phase0 / 50.0, d_phase2 / 50.0, d_phase4 / 50.0, d_phase5 / 50.0, d_phase7 / 50.0, elapsed_green / MAX_GREEN, phase_position]
```

Demand is computed from the halting vehicles on the incoming lanes served by each green phase. The phase-to-lane mapping is built dynamically from the loaded SUMO traffic light logic in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:37).

## Action Space

The action space is defined by `action_dim = 1 + len(GREEN_PHASES)` in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:302).

Actions are:

- action `0`: `EXTEND`
- action `1`: switch to green phase `0`
- action `2`: switch to green phase `2`
- action `3`: switch to green phase `4`
- action `4`: switch to green phase `5`
- action `5`: switch to green phase `7`

This direct target-phase selection is decoded in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:287).

## Timing And Safety Logic

The learned policy is constrained by fixed signal-timing rules from [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:143):

- `EXTEND_STEP = 3`
- `YELLOW_DURATION = 3`
- `MIN_GREEN = 5`
- `MAX_GREEN = 90`

The controller logic is:

- `EXTEND` keeps the active green for 3 more seconds
- a requested switch before `MIN_GREEN` is overridden to extend
- if `MAX_GREEN` is reached, the controller forces a switch to the next green in the nominal cycle
- any switch first applies a generated yellow transition, then applies the target green

## Yellow Transition Design

The controller does not rely on a single fixed yellow phase per source phase anymore. Instead, it synthesizes a transition state from the current and target green phases in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:251).

The rule is:

- if a signal is active now and active in the target green, keep it active during the transition
- if it is active now but not active in the target green, change it to yellow
- if it is not active now, keep it red during the transition

This makes the clearance phase pair-specific, so transitions like `2 -> 4` and `2 -> 7` can produce different yellow states.

## Reward Function

The reward is computed in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:416).

Current reward terms are:

```python
reward = 2.0 * (prev_total_queue - current_total_queue)
reward += 0.2 * (prev_max_lane_queue - current_max_lane_queue)
reward += 1.0 * (arrived_delta)
reward -= 0.1 * (loaded_delta)
reward -= 0.08 * excess_green
if switched:
    reward -= 0.35
reward = reward / 10.0
```

Interpretation:

- reducing total queue is strongly rewarded
- reducing the worst lane queue is weakly rewarded to avoid starvation
- clearing vehicles gives positive reward
- incoming traffic pressure gives a small penalty
- over-holding a green after `MIN_GREEN` is penalized
- each actual switch has a small direct cost

## Network Architecture

The DQN in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:147) is a simple multilayer perceptron:

- input size: `7`
- hidden layer 1: `128`
- hidden layer 2: `128`
- output size: `6`

The 6 outputs correspond to:

- `Q(state, EXTEND)`
- `Q(state, switch->0)`
- `Q(state, switch->2)`
- `Q(state, switch->4)`
- `Q(state, switch->5)`
- `Q(state, switch->7)`

## Training Algorithm

Training uses standard DQN components in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:305):

- replay buffer capacity: `10000`
- batch size: `64`
- discount factor `gamma`: `0.99`
- Adam optimizer
- learning rate: `1e-4`
- target network update frequency: every `5` episodes

Exploration is epsilon-greedy:

- starts at `1.0`
- decays toward `0.05`
- main decay happens over `80` episodes

By default, training can sample multiple deterministic traffic scenarios across episodes.

## What The Agent Learns

The agent learns:

- which movement should receive the next green
- when the current green should be extended
- how to trade off queue reduction, throughput, and worst-case lane pressure

It does not learn:

- arbitrary signal strings
- vehicle routes
- phase geometry
- unconstrained unsafe switching

Those are still bounded by the SUMO network and the controller logic.

## Testing And Evaluation

[test_sim.py](/Users/macbook/Documents/komitas-vagharshyan/test_sim.py:1) runs the trained model in `sumo-gui` for visual inspection. It prints:

- state
- Q-values
- chosen action

[evaluate.py](/Users/macbook/Documents/komitas-vagharshyan/evaluate.py:1) runs the same scenario twice:

- baseline SUMO controller
- RL controller

and compares traffic metrics such as queue, waiting time, trip duration, time loss, and maximum waiting time.

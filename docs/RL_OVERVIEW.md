# RL Overview

This project uses a Deep Q-Network (DQN) to control one traffic light system in SUMO. The agent does not choose arbitrary signal states. It only decides whether to keep the current green phase active a little longer or move to the next green phase in a fixed cycle.

## Problem Setup

- Environment: Eclipse SUMO, accessed through TraCI
- Controlled junction: `cluster_11668441165_11668441166_11668441167_2912634528_#8more`
- Decision maker: a DQN defined in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:61)
- Output model: `dqn_model.pth`

The controller is cycle-based. The allowed green phases are:

- phase `0`
- phase `2`
- phase `4`
- phase `6`

Those are listed in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:19) as `GREEN_PHASES = [0, 2, 4, 6]`.

Intermediate yellow phases are already encoded inside the SUMO network and are used during switching.

## State Representation

The state has four continuous features, created in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:127):

1. `d_current`: demand on the currently active green phase
2. `d_next`: demand on the next green phase in the fixed cycle
3. `d_others`: maximum demand among the other green phases
4. `elapsed_green`: how long the current green phase has been active

The state vector is:

```python
[d_current / 50.0, d_next / 50.0, d_others / 50.0, elapsed_green / 90.0]
```

Demand is computed as the sum of halting vehicles on the incoming lanes associated with a phase. That mapping is built dynamically from SUMO traffic-light logic in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:26).

## Action Space

The action space has size `2`, defined in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:179):

- action `0`: `EXTEND`
- action `1`: `SWITCH`

The agent does not select a target phase directly. If it switches, the code advances to the next green phase in the fixed order:

```text
0 -> 2 -> 4 -> 6 -> 0
```

That logic is implemented in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:99) and [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:157).

## Safety And Timing Logic

The learned action is constrained by traffic-signal rules:

- `EXTEND_STEP = 3` seconds
- `YELLOW_DURATION = 3` seconds
- `MIN_GREEN = 15` seconds
- `MAX_GREEN = 90` seconds

These constants are defined in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:53).

The actual control policy is:

- If the model chooses `EXTEND`, the current green is held for 3 more seconds.
- If the green duration reaches `MAX_GREEN`, the controller forces a switch.
- If the model chooses `SWITCH` before `MIN_GREEN`, the request is overridden and treated like an extend.
- If switching is allowed, the controller enters the corresponding yellow phase first, then activates the next green phase.

This is important: the neural network learns inside a constrained controller, not a fully unconstrained action space.

## Reward Function

The reward is computed in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:251).

First, total congestion is measured as:

```python
0.5 * total_wait + 4.0 * total_queue
```

where:

- `total_wait` is the sum of lane waiting times
- `total_queue` is the sum of halted vehicles

Then reward is:

```python
reward = previous_congestion - current_congestion
excess_green = max(0, elapsed_green - MIN_GREEN)
reward -= 0.15 * excess_green
if action == 1:
    reward -= 1
reward = reward / 100.0
```

Interpretation:

- congestion going down gives positive reward
- greens that remain active too long after the minimum are penalized
- switching has a direct cost
- reward is scaled down for numerical stability

## Network Architecture

The DQN is a simple multilayer perceptron:

- input size: `4`
- hidden layer 1: `128`
- hidden layer 2: `128`
- output size: `2`

Defined in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:61).

The output is two Q-values:

- `Q(state, EXTEND)`
- `Q(state, SWITCH)`

The larger value determines the chosen action during exploitation.

## Training Algorithm

Training uses standard DQN ingredients:

- replay buffer capacity: `10000`
- batch size: `64`
- discount factor `gamma`: `0.99`
- optimizer: Adam
- learning rate: `1e-4`
- target network update frequency: every `5` episodes

These are all in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:182).

Exploration uses epsilon-greedy action selection:

- starts at `1.0`
- decays toward `0.05`
- decay schedule is spread across `80` episodes

Training loop defaults:

- `100` episodes
- `100` decisions per episode

Each training step does:

1. Build current state
2. Choose action with epsilon-greedy DQN
3. Apply control logic in SUMO
4. Observe next congestion
5. Compute reward
6. Store transition in replay memory
7. Sample a minibatch if enough data exists
8. Perform one DQN optimization step

## What The Agent Actually Learns

The agent is learning a high-level phase-duration policy, not low-level traffic engineering from scratch.

More concretely, it learns:

- when the current phase should be extended
- when it is worth paying the switch penalty
- how to balance current demand against near-future demand

It is not learning:

- arbitrary phase ordering
- lane-specific signal states
- direct vehicle routing

Those parts remain fixed by the SUMO network and the controller logic.

## Testing Behavior

Testing is handled by [test_sim.py](/Users/macbook/Documents/komitas-vagharshyan/test_sim.py:9).

At test time:

- the saved model is loaded from `dqn_model.pth`
- epsilon exploration is removed
- the action with the largest Q-value is always chosen
- `MIN_GREEN` and `MAX_GREEN` constraints still apply

The script prints:

- the normalized state
- the model Q-values
- the selected action

That makes it easier to inspect what the model is doing during a SUMO GUI run.

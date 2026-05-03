# RL Overview

This project uses Deep Q-Networks (DQNs) to control a selected set of traffic lights along the Komitas corridor in SUMO. The current setup is multi-intersection, but still locally structured: each controlled traffic light has its own local DQN policy and local state/action space.

## Controlled Intersections

The active controlled TLS set is defined in [train.py](train.py:17):

- `Komitas-Gyulbenkyan`
- `Komitas-Vagharshyan`
- `Komitas-Papazyan`
- `Komitas-Vracakan`
- `Komitas-Griboyedov`
- `Komitas-Tigranyan`

## Agent Structure

The code trains one local DQN per controlled intersection. All of them run in the same SUMO simulation and are updated together during training.

Each DQN is:

- a multilayer perceptron
- hidden size `128 -> 128`
- local input size depends on how many stable green phases that TLS has (see State Representation)
- local output size is `1 + number_of_green_phases`

The first action is always `EXTEND`, and the remaining actions select a target green phase directly.

The networks use **Double DQN** updates: the online policy network selects the next action, and the target network evaluates it. This reduces Q-value overestimation compared to vanilla DQN.

## State Representation

For one traffic light with `k` stable green phases, the local state vector has dimension:

```text
k + 13
```

The components are:

| Index range | Description |
|---|---|
| `0 .. k-1` | Per-phase pressure: normalized halting vehicles on each phase's incoming lanes, divided by 30 |
| `k` | Elapsed green time normalized by `MAX_GREEN` |
| `k+1` | Phase position: index of current green phase in the nominal cycle, normalized to `[0, 1]` |
| `k+2` | Total queue pressure: total halting vehicles divided by `lane_count * 12` |
| `k+3` | Total wait pressure: total lane waiting time divided by `lane_count * 300` |
| `k+4` | Worst-lane pressure: max lane queue divided by 20 |
| `k+5` | Spillback pressure: downstream outgoing queue divided by `outgoing_lane_count * 12` |
| `k+6` | Network traffic load: vehicles currently in the network divided by 500 |
| `k+7` | Previous neighbor: normalized queue pressure |
| `k+8` | Previous neighbor: normalized spillback pressure |
| `k+9` | Previous neighbor: phase position |
| `k+10` | Next neighbor: normalized queue pressure |
| `k+11` | Next neighbor: normalized spillback pressure |
| `k+12` | Next neighbor: phase position |

The neighbor features provide upstream and downstream corridor context for implicit coordination. Terminal intersections (no prev or next) have zero-padded neighbor features.

## Action Space

For one traffic light, the local actions are:

- `0`: `EXTEND` — keep the current green phase active for another `EXTEND_STEP` seconds
- `1..k`: select one of the `k` stable green phases as the target phase

For example, if a traffic light has two green phases, the network has three raw outputs:

```text
0 = EXTEND current green
1 = select green phase 1
2 = select green phase 2
```

If phase 1 is already active, action `1` is masked, so the practical choice is `EXTEND phase 1` or `switch to phase 2`. If phase 2 is active, action `2` is masked, so the practical choice is `EXTEND phase 2` or `switch to phase 1`.

This is stronger than a simple fixed-cycle "switch to next phase" controller because the agent can prioritize the most urgent movement directly.

Invalid actions are masked during training, Double DQN target calculation, evaluation, and GUI visualization. Before the minimum green time is reached, only `EXTEND` is valid. After `MIN_GREEN`, switch actions to the current phase remain invalid, while switch actions to other green phases are available.

## Timing Constraints

The RL policy is constrained by fixed safety timing:

- `EXTEND_STEP = 5` — simulation seconds per decision step
- `YELLOW_DURATION = 3` — yellow intergreen seconds on phase transition
- `MIN_GREEN = 10` — minimum green time before a switch is allowed
- `MAX_GREEN = 90` — maximum green time; forces a switch when exceeded

If a switch would violate `MIN_GREEN`, it is masked out before action selection. If a green overruns `MAX_GREEN`, the controller forces a switch to the next green in the nominal cycle for that TLS.

## Yellow Transition Logic

The controller synthesizes a pair-specific transition state from:

- the current green phase state string
- the target green phase state string

Rule:

- signals active in both states stay active during transition
- signals active now but inactive in the target state go yellow (`y`)
- signals inactive now stay red during transition

This makes transitions pair-specific and is essential for correct direct target-phase control.

## Reward Design

Each controlled intersection receives a local pressure-style reward at every decision step. The current reward combines:

| Term | Purpose |
|---|---|
| Phase pressure penalty | Penalizes unbalanced pressure across phase movements, following the PressLight/MaxPressure idea |
| Local queue penalty | Penalizes halting vehicles on the intersection's controlled lanes |
| Local wait penalty | Penalizes accumulated waiting time |
| Downstream spillback penalty | Penalizes queues on outgoing lanes to avoid blocking downstream intersections |
| Served-pressure bonus | Gives credit for serving a high-pressure current phase |
| Max-lane-wait starvation penalty | Strongly penalizes one approach waiting too long, especially important in low-demand `off_peak` cases |
| Demand-scaled switch penalty | Penalizes switching more strongly when demand is low to reduce yellow-time waste |
| Throughput bonus | Small shared bonus for arrived vehicles |
| Excess-green penalty | Discourages holding a green far beyond `STARVATION_GREEN` |

The reward is divided by `REWARD_SCALE` and further normalized by observed traffic/load so high-demand `corridor_stress` episodes do not dominate low-demand `off_peak` episodes. This scenario-scale normalization was added after earlier runs showed that stress scenarios produced much larger TD targets and weakened generalization.

## Training Structure

During one training step:

1. each TLS builds its local state
2. each TLS chooses a valid local action with epsilon-greedy exploration and action masking
3. all chosen actions are applied together to the network
4. local rewards are computed per TLS
5. each local replay buffer stores its own transition and the valid next-action set
6. each local DQN is updated from its own sampled minibatches using masked Double DQN with gradient clipping (`max_norm=10.0`)

Epsilon decays linearly from `1.0` to `0.05` over the first `85%` of training episodes, then stays at `0.05` for the remaining exploitation-focused episodes.

Training episodes use a balanced round-robin schedule across `corridor_stress`, `evening_rush`, `morning_rush`, and `off_peak` and seeds `41`, `42`, and `43`. So the project is multi-intersection, but the learning remains decentralized at the policy level.

## Testing And Evaluation

- [test_sim.py](test_sim.py:1) runs the trained multi-intersection controller in `sumo-gui`
- [evaluate.py](evaluate.py:1) compares fixed-time, MaxPressure, and RL across the whole controlled corridor
- [batch_evaluate.py](batch_evaluate.py:1) evaluates across multiple scenarios and seeds

## Practical Interpretation

This repository is a corridor-style multi-intersection setup where:

- the network is global (all six TLS run in one simulation)
- rewards are local plus neighbor coordination components plus a shared flow component
- policies are local per TLS (decentralized multi-agent)
- all controlled intersections act simultaneously in the same simulation
- Q-value overestimation is addressed via Double DQN
- invalid actions are masked during action selection and target calculation
- training stability is enforced via gradient clipping

The selected final model, `models/komitas_dqn_ep075.pth`, improves the aggregate batch-evaluation delay metrics versus fixed-time control across four scenarios and three seeds: average queue per step (+9.2%), average trip duration (+5.5%), average waiting time (+5.9%), average time loss (+6.2%), and maximum waiting time (+18.0%). It still has a throughput gap (-6.5% vehicles arrived) and performs worse than fixed-time on `morning_rush`, so the result should be presented as a meaningful but mixed improvement rather than a complete solution.

## Key References

The algorithm and reward design draw from the following sources:

- **Mnih et al. (2015)** — DQN: original experience replay + target network architecture (*Nature*)
- **Van Hasselt et al. (2016)** — Double DQN: reduces Q-value overestimation bias (*AAAI*)
- **Webster (1958)** — classic traffic signal optimisation objective (queue / delay minimisation)
- **Wei et al. (2019, KDD)** — PressLight: pressure reward for arterial network coordination
- **Wei et al. (2019, CIKM)** — CoLight: neighbour-aware multi-intersection MARL
- **Liang et al. (2019, IEEE TITS)** — waiting-time reward in deep RL for traffic signal control
- **Roess et al. (2004)** — Traffic Engineering: signal timing constraints, yellow clearance, green time bounds
- **ITE Handbook (2009)** — green-time equity, minimum green requirements
- **Lopez et al. (2018, ITSC)** — SUMO microscopic traffic simulation platform

# Project Report

## 1. Project Objective

This project studies reinforcement-learning-based traffic signal control for the Komitas-Vagharshyan intersection in SUMO. The goal is to improve traffic efficiency over the default fixed SUMO controller while preserving safe signal timing behavior.

The final system is a hybrid controller:

- SUMO provides the road network, traffic light geometry, and vehicle simulation
- rule-based logic enforces minimum green time, maximum green time, and yellow clearance
- a Deep Q-Network (DQN) chooses whether to extend the current green or switch to a specific valid green phase

This means the project does not let the agent emit arbitrary signal strings. The agent learns high-level phase control inside a constrained traffic-engineering envelope.

## 2. Environment And Assets

The main simulation assets are stored in [sumo_data](/Users/macbook/Documents/komitas-vagharshyan/sumo_data):

- [komitas-vagharshyan.net.xml](/Users/macbook/Documents/komitas-vagharshyan/sumo_data/komitas-vagharshyan.net.xml:1): road network and traffic light program
- [komitas-vagharshyan.sumocfg](/Users/macbook/Documents/komitas-vagharshyan/sumo_data/komitas-vagharshyan.sumocfg:1): SUMO scenario configuration
- [routes.rou.xml](/Users/macbook/Documents/komitas-vagharshyan/sumo_data/routes.rou.xml:1): active route demand file
- [komitas-vagharshyan.osm](/Users/macbook/Documents/komitas-vagharshyan/sumo_data/komitas-vagharshyan.osm:1): raw map source

The code controls one traffic light:

- `cluster_11668441165_11668441166_11668441167_2912634528_#8more`

## 3. Reinforcement Learning Formulation

### 3.1 Agent

The agent is a DQN implemented in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:136). It uses a feed-forward neural network with:

- input dimension: `7`
- two hidden layers of size `128`
- output dimension: `6`

The 6 outputs are Q-values for:

- `EXTEND`
- `SWITCH -> phase 0`
- `SWITCH -> phase 2`
- `SWITCH -> phase 4`
- `SWITCH -> phase 5`
- `SWITCH -> phase 7`

### 3.2 Controlled Green Phases

The currently valid green phases are defined in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:20):

- `0`
- `2`
- `4`
- `5`
- `7`

The nominal cyclic order used for forced switching is defined in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:21):

- `0 -> 2 -> 4 -> 5 -> 7 -> 0`

### 3.3 State Definition

The state is built in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:194). It contains:

1. normalized demand for phase `0`
2. normalized demand for phase `2`
3. normalized demand for phase `4`
4. normalized demand for phase `5`
5. normalized demand for phase `7`
6. normalized elapsed green time
7. normalized phase position

The exact state vector is:

```python
[d_phase0 / 50.0, d_phase2 / 50.0, d_phase4 / 50.0, d_phase5 / 50.0, d_phase7 / 50.0, elapsed_green / MAX_GREEN, phase_position]
```

Demand for a phase is computed from lane-level queue observations:

- the code dynamically maps each green phase to its incoming lanes using the loaded SUMO traffic light logic
- phase demand is the sum of halting vehicles on those incoming lanes

This gives the agent a compact phase-level view of intersection pressure while still grounding it in lane-level traffic measurements.

### 3.4 Action Definition

The action encoding is defined in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:274):

- action `0`: `EXTEND`
- action `1`: switch to phase `0`
- action `2`: switch to phase `2`
- action `3`: switch to phase `4`
- action `4`: switch to phase `5`
- action `5`: switch to phase `7`

This is a direct target-phase controller, not a simple “switch to next phase” controller.

That is an important design choice. Earlier simpler controllers can only decide whether to keep the current green or move to the next fixed phase. The current controller is stronger because it can prioritize the most urgent green phase directly.

## 4. Signal Timing Logic

The RL policy is constrained by fixed safety and timing rules from [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:127):

- `EXTEND_STEP = 3`
- `YELLOW_DURATION = 3`
- `MIN_GREEN = 5`
- `MAX_GREEN = 90`

The control rules are:

- if the chosen action is `EXTEND`, the green is held for 3 more seconds
- if the model requests a switch before `MIN_GREEN`, the request is overridden and treated as an extend
- if the green duration reaches `MAX_GREEN`, the controller forces a switch to the next green in the nominal cycle

This keeps the learned policy inside a safe operating envelope.

## 5. Yellow Transition Design

The current implementation does not rely on one fixed yellow phase per source phase. Instead, it synthesizes a pair-specific transition state in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:231).

Given a current green state and a target green state:

- movements active in both states remain active during transition
- movements active now but not active in the target state become yellow
- movements not active now remain red during transition

This is important because yellow clearance depends on the actual transition pair. A switch from one green phase to another may need a different transition pattern than a switch to a third phase.

After the generated yellow transition is applied for `YELLOW_DURATION`, the controller applies the target green state directly.

## 6. Reward Design

The reward is computed in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:390). The current reward is:

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

- queue reduction is the main objective
- a small worst-lane term discourages starvation on one approach
- throughput is rewarded through newly arrived vehicles
- incoming traffic pressure is lightly penalized through newly loaded vehicles
- holding green too long after the minimum is penalized
- every actual switch has a small direct cost

The reward intentionally does not optimize only one metric. It tries to balance:

- average flow quality
- queue pressure
- worst-case lane protection
- throughput
- switching stability

## 7. Training Process

Training is implemented in [train.py](/Users/macbook/Documents/komitas-vagharshyan/train.py:281).

### 7.1 DQN Setup

Training uses:

- replay buffer capacity: `10000`
- batch size: `64`
- discount factor `gamma = 0.99`
- Adam optimizer
- learning rate `1e-4`
- target network update frequency: every `5` episodes

Exploration is epsilon-greedy:

- starts at `1.0`
- decays toward `0.05`
- main decay spans `80` episodes

### 7.2 Episode Structure

Each training episode does the following:

1. choose a route file
2. create a runnable SUMO config
3. start SUMO headlessly
4. initialize the phase-to-lane mapping
5. warm up the simulation for 20 steps
6. repeat the decision loop for `decisions_per_episode`

The decision loop is:

1. read the current state
2. choose an action with epsilon-greedy DQN
3. apply controller rules and phase transition logic
4. observe the next traffic metrics
5. compute reward
6. store the transition in replay memory
7. sample a minibatch if enough transitions exist
8. update the policy network

At the end of each episode:

- the target network is updated periodically
- checkpoints may be saved
- the final model is saved to `dqn_model.pth`

### 7.3 Training Scenarios

Training can use:

- one fixed route file
- or multiple generated scenarios across episodes

The multi-scenario path uses deterministic route generation from [scripts/generate_traffic.py](/Users/macbook/Documents/komitas-vagharshyan/scripts/generate_traffic.py:1) and stores generated routes under `training_routes/`.

This design is useful because:

- one fixed route is easier for debugging
- multiple route scenarios improve robustness and reduce overfitting

## 8. Traffic Scenarios

Scenario definitions live in [scripts/generate_traffic.py](/Users/macbook/Documents/komitas-vagharshyan/scripts/generate_traffic.py:5).

The built-in scenarios are:

- `morning_rush`
- `evening_rush`
- `off_peak`

Each scenario is a departure-probability profile over source edges. A route file is generated by:

1. choosing a scenario
2. choosing a random seed
3. iterating over simulation time steps
4. creating vehicles probabilistically according to the scenario’s edge probabilities

This means:

- same scenario + same seed gives the same route file
- same scenario + different seed gives a similar but not identical traffic realization

## 9. Testing Process

Testing for visual inspection is implemented in [test_sim.py](/Users/macbook/Documents/komitas-vagharshyan/test_sim.py:1).

Its purpose is qualitative rather than statistical. It:

- loads the trained model
- starts `sumo-gui`
- runs the trained controller greedily with no exploration
- prints state, Q-values, and chosen action

During testing:

- the model is not updated
- replay memory is not used
- only inference is performed

This script is useful for:

- watching signal behavior
- understanding policy choices
- checking whether switches happen sensibly
- catching obvious controller regressions

## 10. Evaluation Process

Quantitative comparison is implemented in [evaluate.py](/Users/macbook/Documents/komitas-vagharshyan/evaluate.py:1).

It compares:

- the default SUMO traffic light controller
- the trained RL controller

For each run, it reports:

- steps
- vehicles arrived
- average queue per step
- average lane wait per step
- average trip duration
- average waiting time
- average time loss
- maximum waiting time

This script is the main baseline-vs-RL comparison tool for one scenario at a time.

## 11. Batch Evaluation

For stronger testing across multiple scenarios and seeds, the project now includes [batch_evaluate.py](/Users/macbook/Documents/komitas-vagharshyan/batch_evaluate.py:1).

This script:

1. generates route files for a scenario/seed grid
2. runs baseline and RL on each one
3. prints a per-route summary
4. prints an averaged comparison table
5. can optionally save the full summary to JSON

This is a better final-evaluation path than relying on one fixed route file.

## 12. Checkpoint-Based Final Training

[run_final_training.py](/Users/macbook/Documents/komitas-vagharshyan/run_final_training.py:1) automates a stronger final selection workflow:

1. train for many episodes
2. save periodic checkpoints
3. generate evaluation scenarios
4. evaluate every checkpoint across those scenarios
5. rank checkpoints by average score
6. write JSON and Markdown summaries

This matters because in RL the final episode is not guaranteed to produce the best-performing model.

## 13. Current Strengths And Limitations

### Strengths

- direct target-phase control gives the agent meaningful authority
- pair-specific yellow synthesis is more realistic than one fixed yellow per source phase
- the reward balances queue, throughput, and starvation prevention
- the project supports both quick single-route evaluation and broader multi-scenario evaluation

### Limitations

- the controller still operates at the phase level, not lane-by-lane signal design
- performance can still vary by traffic scenario and seed
- the generated yellow transition is rule-based and generic, not calibrated from detailed signal-engineering standards
- the route-generation scenarios are simplified demand models, not full real-world time-of-day calibration

## 14. Recommended Usage

For normal development:

1. train on one fixed route or a small scenario set
2. inspect behavior with `test_sim.py`
3. compare against baseline with `evaluate.py`

For stronger final evaluation:

1. train with checkpoints using `run_final_training.py`
2. evaluate across multiple scenarios and seeds using `batch_evaluate.py`
3. report averaged results, not just one fixed-route comparison

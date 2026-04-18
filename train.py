import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci

SUMO_CONFIG = "sumo_data/komitas-vagharshyan.sumocfg"
SUMO_BINARY = os.environ.get("SUMO_BINARY", "sumo-gui")
SUMO_CMD = [SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--quit-on-end", "--no-warnings"]
TLS_ID = "cluster_11668441165_11668441166_11668441167_2912634528_#8more"
MODEL_PATH = "dqn_model.pth"
RANDOM_SEED = 42

GREEN_PHASES = [0, 2, 4, 6]

# -------------------------------
# Phase -> lane mapping
# -------------------------------
PHASE_LANES = {}

def init_phase_lanes():
    """Dynamically build the phase -> lane mapping from the loaded SUMO network."""
    global PHASE_LANES
    PHASE_LANES.clear()
    
    logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
    controlled_links = traci.trafficlight.getControlledLinks(TLS_ID)
    
    for phase_idx in GREEN_PHASES:
        state = logic.phases[phase_idx].state
        active_in_lanes = set()
        
        for link_idx, signal_char in enumerate(state):
            # 'G', 'g' and 's' represent active signals
            if signal_char in ("G", "g", "s"):
                if link_idx < len(controlled_links):
                    links = controlled_links[link_idx]
                    if links:
                        # connections are tuples: (incoming_lane, outgoing_lane, via_lane)
                        for connection in links:
                            active_in_lanes.add(connection[0])
                            
        PHASE_LANES[phase_idx] = list(active_in_lanes)

# -------------------------------
# PARAMETERS
# -------------------------------
EXTEND_STEP = 3
YELLOW_DURATION = 3
MIN_GREEN = 15
MAX_GREEN = 90

# -------------------------------
# DQN CLASSES
# -------------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# -------------------------------
# HELPERS
# -------------------------------
def step_for_seconds(seconds, render_delay=0.0):
    """Advance the simulation by a fixed number of seconds."""
    for _ in range(seconds):
        traci.simulationStep()
        if render_delay > 0.0:
            time.sleep(render_delay)

def get_next_phase(current):
    """Returns the next green phase in the fixed cycle GREEN_PHASES."""
    idx = GREEN_PHASES.index(current)
    return GREEN_PHASES[(idx + 1) % len(GREEN_PHASES)]

def get_other_phases(current):
    """Returns all green phases except the current one and the immediate next one."""
    next_p = get_next_phase(current)
    return [p for p in GREEN_PHASES if p not in (current, next_p)]

# -------------------------------
# DEMAND
# -------------------------------
def get_phase_demand(phase):
    """
    Demand is the sum of halting vehicles on unique lanes belonging to a given phase.
    """
    lanes = set(PHASE_LANES[phase])
    return sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

# -------------------------------
# STATE
# -------------------------------
# State meaning (CONTINUOUS NOW for DQN):
# d_current: raw demand of the currently active green phase
# d_next: raw demand of the next expected phase in cycle
# d_others: max raw demand among the remaining phases
# elapsed_green: duration the current phase has been green
def get_state(current_phase, elapsed_green):
    next_phase = get_next_phase(current_phase)
    other_phases = get_other_phases(current_phase)

    d_current = get_phase_demand(current_phase)
    d_next = get_phase_demand(next_phase)
    d_others = max((get_phase_demand(p) for p in other_phases), default=0)

    # Normalize state variables for neural network stability.
    # Demand is scaled by an assumed 50 vehicles per phase and elapsed green by 90s.
    return np.array(
        [d_current / 50.0, d_next / 50.0, d_others / 50.0, elapsed_green / 90.0],
        dtype=np.float32,
    )

# -------------------------------
# REWARD & CONGESTION
# -------------------------------
def get_total_congestion():
    """Computes total congestion across all lanes controlled by the TLS."""
    lanes = list(set(traci.trafficlight.getControlledLanes(TLS_ID)))
    
    total_wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
    total_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

    return 0.5 * total_wait + 4.0 * total_queue

# -------------------------------
# SWITCH LOGIC
# -------------------------------
def switch_phase(current_phase, render_delay=0.0):
    """Safely transitions to the next phase as defined entirely by the SUMO network logic."""
    next_phase = get_next_phase(current_phase)

    # Rely precisely on the .net.xml predefined phase sequence.
    # Assumes (current_phase + 1) is the transition phase set up in SUMO.
    traci.trafficlight.setPhase(TLS_ID, current_phase + 1)
    step_for_seconds(YELLOW_DURATION, render_delay=render_delay)
    traci.trafficlight.setPhase(TLS_ID, next_phase)

    return next_phase

# -------------------------------
# TRAIN
# -------------------------------
def train():
    """Train the DQN traffic light controller and save it to disk."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    state_dim = 4
    action_dim = 2  # extend (0) / switch (1)

    # Hyperparameters
    batch_size = 64
    gamma = 0.99
    lr = 1e-4
    target_update_freq = 5

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(10000)

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 80  # Decay much slower across more episodes

    episodes = 100
    decisions_per_episode = 100
    loss_fn = nn.MSELoss()

    for episode in range(episodes):
        traci.start(SUMO_CMD)
        try:
            if not PHASE_LANES:
                init_phase_lanes()

            current_phase = traci.trafficlight.getPhase(TLS_ID)
            elapsed_green = 0

            # Stabilize simulation slightly
            step_for_seconds(20)

            episode_reward = 0
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
                0, (epsilon_decay - episode) / epsilon_decay
            )

            for step in range(decisions_per_episode):
                state = get_state(current_phase, elapsed_green)

                # Epsilon-greedy action
                if random.random() < epsilon:
                    action = random.randint(0, 1)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        action = policy_net(state_tensor).argmax().item()

                prev_congestion = get_total_congestion()

                if action == 0:  # Extend
                    step_for_seconds(EXTEND_STEP)
                    elapsed_green += EXTEND_STEP

                    if elapsed_green >= MAX_GREEN:
                        current_phase = switch_phase(current_phase)
                        elapsed_green = 0
                else:  # Switch
                    if elapsed_green < MIN_GREEN:
                        step_for_seconds(EXTEND_STEP)
                        elapsed_green += EXTEND_STEP
                    else:
                        current_phase = switch_phase(current_phase)
                        elapsed_green = 0
                        step_for_seconds(EXTEND_STEP)

                current_congestion = get_total_congestion()

                reward = prev_congestion - current_congestion

                # After the minimum green has been satisfied, progressively penalize
                # holding the phase too long so the agent does not collapse to EXTEND.
                excess_green = max(0, elapsed_green - MIN_GREEN)
                reward -= 0.15 * excess_green

                if action == 1:
                    reward -= 1

                # Scale reward down to keep gradients stable.
                reward = reward / 100.0

                next_state = get_state(current_phase, elapsed_green)
                done = step == (decisions_per_episode - 1)

                memory.push(state, action, reward, next_state, done)
                episode_reward += reward

                if len(memory) >= batch_size:
                    states, actions, rewards, next_states, dones = memory.sample(batch_size)

                    states = torch.FloatTensor(states)
                    actions = torch.LongTensor(actions).unsqueeze(1)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1)
                    next_states = torch.FloatTensor(next_states)
                    dones = torch.FloatTensor(dones).unsqueeze(1)

                    q_values = policy_net(states).gather(1, actions)
                    with torch.no_grad():
                        next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                        target_q_values = rewards + (gamma * next_q_values * (1 - dones))

                    loss = loss_fn(q_values, target_q_values)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        finally:
            traci.close()

        # Update Target Network
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode+1}/{episodes}: total reward = {episode_reward:.2f} | epsilon = {epsilon:.2f}")

    torch.save(policy_net.state_dict(), MODEL_PATH)
    print(f"Training complete! Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()

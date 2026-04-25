import argparse
import os
import random
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci
from scripts.generate_traffic import SCENARIOS, generate_route_file

SUMO_CONFIG = "sumo_data/komitas-vagharshyan.sumocfg"
SUMO_BINARY = os.environ.get("SUMO_BINARY", "sumo")
TLS_ID = "cluster_11668441165_11668441166_11668441167_2912634528_#8more"
MODEL_PATH = "dqn_model.pth"
RANDOM_SEED = 42
DEFAULT_TRAINING_SCENARIOS = sorted(SCENARIOS)
DEFAULT_TRAINING_SEEDS = [41, 42, 43]

GREEN_PHASES = [0, 2, 4, 5, 7]
PHASE_TRANSITIONS = {
    0: {"next_green": 2},
    2: {"next_green": 4},
    4: {"next_green": 5},
    5: {"next_green": 7},
    7: {"next_green": 0},
}
ACTION_EXTEND = 0
STATE_DIM = len(GREEN_PHASES) + 2
ACTION_DIM = 1 + len(GREEN_PHASES)

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


def create_config_with_route_override(route_file=None):
    """Create a SUMO config, optionally overriding the route file path."""
    if route_file is None:
        return SUMO_CONFIG, None

    tree = ET.parse(SUMO_CONFIG)
    root = tree.getroot()
    config_dir = Path(SUMO_CONFIG).resolve().parent
    input_node = root.find("./input")
    if input_node is None:
        input_node = ET.SubElement(root, "input")

    # When the temporary config is written outside the repo, SUMO resolves any
    # relative asset path against that temp directory. Rewrite known file inputs
    # to absolute paths so the copied config remains runnable.
    for tag in ("net-file", "route-files", "additional-files"):
        node = root.find(f"./input/{tag}")
        if node is None:
            continue
        value = node.get("value")
        if not value:
            continue
        parts = [part.strip() for part in value.split(",")]
        abs_parts = []
        for part in parts:
            part_path = Path(part)
            if not part_path.is_absolute():
                part_path = (config_dir / part_path).resolve()
            abs_parts.append(str(part_path))
        node.set("value", ",".join(abs_parts))

    route_node = root.find("./input/route-files")
    if route_node is None:
        route_node = ET.SubElement(input_node, "route-files")

    route_node.set("value", str(Path(route_file).resolve()))
    tmp = tempfile.NamedTemporaryFile(suffix=".sumocfg", delete=False)
    tmp.close()
    tree.write(tmp.name, encoding="utf-8", xml_declaration=True)
    return tmp.name, tmp.name


def build_sumo_cmd(config_path, sumo_binary=None):
    """Build the SUMO command used for training or testing."""
    binary = sumo_binary or SUMO_BINARY
    cmd = [binary, "-c", config_path, "--quit-on-end", "--no-warnings"]
    if binary.endswith("sumo-gui"):
        cmd.insert(-2, "--start")
    return cmd


def prepare_training_route_files(scenarios, seeds, steps, output_dir):
    """Generate deterministic route files used during multi-scenario training."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    route_files = []

    for scenario in scenarios:
        for seed in seeds:
            route_file = output_path / f"{scenario}_seed{seed}.rou.xml"
            if not route_file.exists():
                generate_route_file(
                    route_file, SCENARIOS[scenario], seed=seed, n_steps=steps
                )
            route_files.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "route_file": str(route_file),
                }
            )

    return route_files


# -------------------------------
# PARAMETERS
# -------------------------------
EXTEND_STEP = 3
YELLOW_DURATION = 3
MIN_GREEN = 5
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
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

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


# -------------------------------
# DEMAND
# -------------------------------
def get_phase_demand(phase):
    """
    Demand is the sum of halting vehicles on unique lanes belonging to a given phase.
    """
    lanes = set(PHASE_LANES[phase])
    return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)


# -------------------------------
# STATE
# -------------------------------
def get_state(current_phase, elapsed_green):
    phase_demands = [get_phase_demand(phase) / 50.0 for phase in GREEN_PHASES]
    phase_position = GREEN_PHASES.index(current_phase) / (len(GREEN_PHASES) - 1)

    # State includes all green-phase demands, elapsed green, and the current
    # phase position in the fixed cycle.
    return np.array(
        phase_demands + [elapsed_green / MAX_GREEN, phase_position],
        dtype=np.float32,
    )


# -------------------------------
# REWARD & CONGESTION
# -------------------------------
def get_congestion_metrics():
    """Compute waiting, queue, and worst-lane pressure across the controlled lanes."""
    lanes = list(set(traci.trafficlight.getControlledLanes(TLS_ID)))
    lane_queues = [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes]
    return {
        "total_wait": sum(traci.lane.getWaitingTime(lane) for lane in lanes),
        "total_queue": sum(lane_queues),
        "max_lane_queue": max(lane_queues, default=0),
        "arrived": traci.simulation.getArrivedNumber(),
        "loaded": traci.simulation.getLoadedNumber(),
    }


# -------------------------------
# SWITCH LOGIC
# -------------------------------
def switch_phase(current_phase, render_delay=0.0):
    """Transition to the next green phase using a synthesized yellow phase."""
    next_phase = PHASE_TRANSITIONS[current_phase]["next_green"]
    return switch_to_phase(current_phase, next_phase, render_delay=render_delay)


def build_transition_state(current_phase, target_phase):
    """Build a yellow transition state from the current and target green phases."""
    logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
    current_state = logic.phases[current_phase].state
    target_state = logic.phases[target_phase].state
    active_signals = {"G", "g", "s"}

    transition = []
    for current_signal, target_signal in zip(current_state, target_state):
        if current_signal in active_signals and target_signal in active_signals:
            transition.append(current_signal)
        elif current_signal in active_signals and target_signal not in active_signals:
            transition.append("y")
        else:
            transition.append("r")

    return "".join(transition)


def apply_green_phase(phase):
    """Apply a named green phase by its state string instead of phase index."""
    logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
    traci.trafficlight.setRedYellowGreenState(TLS_ID, logic.phases[phase].state)


def switch_to_phase(current_phase, target_phase, render_delay=0.0):
    """Switch from the current green to a target green using a pair-specific yellow."""
    transition_state = build_transition_state(current_phase, target_phase)
    if transition_state != traci.trafficlight.getRedYellowGreenState(TLS_ID):
        traci.trafficlight.setRedYellowGreenState(TLS_ID, transition_state)
        step_for_seconds(YELLOW_DURATION, render_delay=render_delay)

    apply_green_phase(target_phase)
    return target_phase


def decode_action(action_id):
    """Map an action id to either EXTEND or a target green phase."""
    if action_id == ACTION_EXTEND:
        return {"type": "extend", "target_phase": None}
    if not 0 <= action_id < ACTION_DIM:
        raise ValueError(
            f"Unknown action id {action_id}. Expected range [0, {ACTION_DIM - 1}]."
        )
    target_phase = GREEN_PHASES[action_id - 1]
    return {"type": "switch", "target_phase": target_phase}


# -------------------------------
# TRAIN
# -------------------------------
def train(
    episodes=100,
    decisions_per_episode=100,
    model_path=MODEL_PATH,
    checkpoint_every=None,
    checkpoint_dir="checkpoints",
    route_file=None,
    training_scenarios=None,
    training_seeds=None,
    training_steps=3600,
    training_route_dir="training_routes",
):
    """Train the DQN traffic light controller and save it to disk."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    training_routes = None
    if route_file is None:
        training_routes = prepare_training_route_files(
            scenarios=training_scenarios or DEFAULT_TRAINING_SCENARIOS,
            seeds=training_seeds or DEFAULT_TRAINING_SEEDS,
            steps=training_steps,
            output_dir=training_route_dir,
        )

    state_dim = STATE_DIM
    action_dim = ACTION_DIM

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

    loss_fn = nn.MSELoss()

    checkpoint_path = None
    if checkpoint_every:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    for episode in range(episodes):
        episode_route = route_file
        episode_label = "fixed route"
        if training_routes is not None:
            selected_route = random.choice(training_routes)
            episode_route = selected_route["route_file"]
            episode_label = (
                f"{selected_route['scenario']} seed {selected_route['seed']}"
            )

        config_path, temp_config_path = create_config_with_route_override(episode_route)
        sumo_cmd = build_sumo_cmd(config_path)
        try:
            traci.start(sumo_cmd)
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
                        action = random.randint(0, action_dim - 1)
                    else:
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0)
                            action = policy_net(state_tensor).argmax().item()

                    prev_metrics = get_congestion_metrics()
                    action_info = decode_action(action)
                    switched = False

                    if action_info["type"] == "extend":
                        step_for_seconds(EXTEND_STEP)
                        elapsed_green += EXTEND_STEP

                        if elapsed_green >= MAX_GREEN:
                            current_phase = switch_phase(current_phase)
                            elapsed_green = 0
                    else:
                        target_phase = action_info["target_phase"]
                        if target_phase == current_phase:
                            step_for_seconds(EXTEND_STEP)
                            elapsed_green += EXTEND_STEP
                        elif elapsed_green < MIN_GREEN:
                            step_for_seconds(EXTEND_STEP)
                            elapsed_green += EXTEND_STEP
                        else:
                            current_phase = switch_to_phase(current_phase, target_phase)
                            elapsed_green = 0
                            switched = True
                            step_for_seconds(EXTEND_STEP)

                    current_metrics = get_congestion_metrics()

                    reward = 2.0 * (
                        prev_metrics["total_queue"] - current_metrics["total_queue"]
                    )
                    reward += 0.2 * (
                        prev_metrics["max_lane_queue"]
                        - current_metrics["max_lane_queue"]
                    )
                    reward += 1.0 * (
                        current_metrics["arrived"] - prev_metrics["arrived"]
                    )
                    reward -= 0.1 * (current_metrics["loaded"] - prev_metrics["loaded"])

                    # After the minimum green has been satisfied, progressively penalize
                    # holding the phase too long so the agent does not collapse to EXTEND.
                    excess_green = max(0, elapsed_green - MIN_GREEN)
                    reward -= 0.08 * excess_green

                    if switched:
                        reward -= 0.35

                    # Scale reward down to keep gradients stable.
                    reward = reward / 10.0

                    next_state = get_state(current_phase, elapsed_green)
                    done = step == (decisions_per_episode - 1)

                    memory.push(state, action, reward, next_state, done)
                    episode_reward += reward

                    if len(memory) >= batch_size:
                        states, actions, rewards, next_states, dones = memory.sample(
                            batch_size
                        )

                        states = torch.FloatTensor(states)
                        actions = torch.LongTensor(actions).unsqueeze(1)
                        rewards = torch.FloatTensor(rewards).unsqueeze(1)
                        next_states = torch.FloatTensor(next_states)
                        dones = torch.FloatTensor(dones).unsqueeze(1)

                        q_values = policy_net(states).gather(1, actions)
                        with torch.no_grad():
                            next_q_values = (
                                target_net(next_states).max(1)[0].unsqueeze(1)
                            )
                            target_q_values = rewards + (
                                gamma * next_q_values * (1 - dones)
                            )

                        loss = loss_fn(q_values, target_q_values)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            finally:
                traci.close()
        finally:
            if temp_config_path:
                os.unlink(temp_config_path)

        # Update Target Network
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(
            f"Episode {episode+1}/{episodes}: total reward = {episode_reward:.2f} | "
            f"epsilon = {epsilon:.2f} | route = {episode_label}"
        )

        if checkpoint_path and (episode + 1) % checkpoint_every == 0:
            checkpoint_file = checkpoint_path / f"checkpoint_ep{episode + 1:03d}.pth"
            torch.save(policy_net.state_dict(), checkpoint_file)
            print(f"Saved checkpoint to {checkpoint_file}")

    torch.save(policy_net.state_dict(), model_path)
    print(f"Training complete! Model saved to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the DQN traffic light controller."
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of training episodes."
    )
    parser.add_argument(
        "--decisions-per-episode",
        type=int,
        default=100,
        help="Number of control decisions per training episode.",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to save the final trained model.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a model checkpoint every N episodes. Disabled when set to 0.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory used for periodic checkpoints.",
    )
    parser.add_argument(
        "--route-file",
        default=None,
        help="Optional route file override for training.",
    )
    parser.add_argument(
        "--training-scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=DEFAULT_TRAINING_SCENARIOS,
        help="Scenario names used for per-episode multi-scenario training.",
    )
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_TRAINING_SEEDS,
        help="Random seeds used to generate training route files.",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=3600,
        help="Simulation length used for generated training route files.",
    )
    parser.add_argument(
        "--training-route-dir",
        default="training_routes",
        help="Directory used for generated training route files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        episodes=args.episodes,
        decisions_per_episode=args.decisions_per_episode,
        model_path=args.model_path,
        checkpoint_every=args.checkpoint_every or None,
        checkpoint_dir=args.checkpoint_dir,
        route_file=args.route_file,
        training_scenarios=args.training_scenarios,
        training_seeds=args.training_seeds,
        training_steps=args.training_steps,
        training_route_dir=args.training_route_dir,
    )

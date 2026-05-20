import argparse
import os
import random
import socket
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci

from scripts.generate_traffic import SCENARIOS, generate_route_file

SUMO_CONFIG = "data/raw_data/sumo_data/komitas.sumocfg"
SUMO_BINARY = os.environ.get("SUMO_BINARY", "sumo")
CONTROLLED_TLS_IDS = [
    "Komitas-Gyulbenkyan",
    "Komitas-Vagharshyan",
    "Komitas-Papazyan",
    "Komitas-Vracakan",
    "Komitas-Griboyedov",
    "Komitas-Tigranyan",
]
NEIGHBOR_TLS_IDS = {
    tls_id: {
        "prev": CONTROLLED_TLS_IDS[idx - 1] if idx > 0 else None,
        "next": CONTROLLED_TLS_IDS[idx + 1]
        if idx < len(CONTROLLED_TLS_IDS) - 1
        else None,
    }
    for idx, tls_id in enumerate(CONTROLLED_TLS_IDS)
}
TLS_ID = CONTROLLED_TLS_IDS[0]
MODEL_PATH = "dqn_model.pth"
RANDOM_SEED = 42
REPORTED_SCENARIOS = ["rush_hour", "off_peak", "corridor_stress"]
DEFAULT_TRAINING_SCENARIOS = REPORTED_SCENARIOS
DEFAULT_TRAINING_SEEDS = [41, 42, 43]
ACTION_EXTEND = 0
REWARD_SCALE = 10.0
EPSILON_DECAY_RATIO = 0.85
LR = 3e-4
REPLAY_BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 5
SWITCH_PENALTY = 0.15
STARVATION_GREEN = 45

EXTEND_STEP = 5
YELLOW_DURATION = 3
MIN_GREEN = 10
MAX_GREEN = 90


def resolve_net_file(config_path):
    """Resolve the net-file path referenced by a SUMO config."""
    config_file = Path(config_path).resolve()
    root = ET.parse(config_file).getroot()
    net_node = root.find("./input/net-file")
    if net_node is None or not net_node.get("value"):
        raise ValueError(f"No <net-file> entry found in {config_path}.")
    net_path = Path(net_node.get("value"))
    if not net_path.is_absolute():
        net_path = (config_file.parent / net_path).resolve()
    return net_path


def is_green_phase(state):
    """Return True when the phase is a stable green serving movements."""
    active_signals = {"G", "g", "s"}
    return "y" not in state and any(signal in active_signals for signal in state)


def load_tls_configs(config_path=SUMO_CONFIG):
    """Parse the SUMO network and derive the controllable green phases per TLS."""
    net_path = resolve_net_file(config_path)
    root = ET.parse(net_path).getroot()
    tls_configs = {}

    for tl_logic in root.findall("tlLogic"):
        tls_id = tl_logic.attrib["id"]
        if tls_id not in CONTROLLED_TLS_IDS:
            continue

        phases = tl_logic.findall("phase")
        phase_states = {idx: phase.attrib["state"] for idx, phase in enumerate(phases)}
        green_phases = [
            idx
            for idx, phase in enumerate(phases)
            if is_green_phase(phase.attrib["state"])
        ]
        if not green_phases:
            raise ValueError(f"TLS {tls_id} has no green phases in {net_path}.")

        next_green = {
            phase: green_phases[(idx + 1) % len(green_phases)]
            for idx, phase in enumerate(green_phases)
        }
        tls_configs[tls_id] = {
            "phase_states": phase_states,
            "green_phases": green_phases,
            "next_green": next_green,
        }

    missing = [tls_id for tls_id in CONTROLLED_TLS_IDS if tls_id not in tls_configs]
    if missing:
        raise ValueError(
            f"Missing tlLogic definitions for controlled TLS IDs: {missing}"
        )

    return tls_configs


TLS_CONFIGS = load_tls_configs(SUMO_CONFIG)
STATE_DIMS = {
    tls_id: len(config["green_phases"]) + 13 for tls_id, config in TLS_CONFIGS.items()
}
ACTION_DIMS = {
    tls_id: 1 + len(config["green_phases"]) for tls_id, config in TLS_CONFIGS.items()
}

# Backward-compatible aliases that some helper scripts and docs may still import.
GREEN_PHASES = TLS_CONFIGS[TLS_ID]["green_phases"]
PHASE_TRANSITIONS = {
    phase: {"next_green": next_phase}
    for phase, next_phase in TLS_CONFIGS[TLS_ID]["next_green"].items()
}
STATE_DIM = STATE_DIMS[TLS_ID]
ACTION_DIM = ACTION_DIMS[TLS_ID]

PHASE_LANES = {}
PHASE_OUT_LANES = {}


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
    cmd = [binary, "-c", config_path, "--quit-on-end", "--no-warnings", "--no-step-log"]
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


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
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

    def push(self, state, action, reward, next_state, done, next_valid_actions):
        self.buffer.append(
            (state, action, reward, next_state, done, next_valid_actions)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_valid_actions = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            list(next_valid_actions),
        )

    def __len__(self):
        return len(self.buffer)


def step_for_seconds(seconds, render_delay=0.0):
    """Advance the simulation by a fixed number of seconds."""
    for _ in range(seconds):
        traci.simulationStep()
        if render_delay > 0.0:
            time.sleep(render_delay)


def start_traci(sumo_cmd):
    """Start TraCI with an explicitly reserved local port."""
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        traci.start(sumo_cmd, port=port)
    except PermissionError:
        traci.start(sumo_cmd)


def init_phase_lanes():
    """Build the phase-to-lane mapping for every controlled TLS."""
    global PHASE_LANES, PHASE_OUT_LANES
    PHASE_LANES = {}
    PHASE_OUT_LANES = {}

    for tls_id in CONTROLLED_TLS_IDS:
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        phase_in_map = {}
        phase_out_map = {}

        for phase_idx in TLS_CONFIGS[tls_id]["green_phases"]:
            state = logic.phases[phase_idx].state
            active_in_lanes = set()
            active_out_lanes = set()

            for link_idx, signal_char in enumerate(state):
                if signal_char in ("G", "g", "s"):
                    if link_idx < len(controlled_links):
                        links = controlled_links[link_idx]
                        if links:
                            for connection in links:
                                active_in_lanes.add(connection[0])
                                active_out_lanes.add(connection[1])

            phase_in_map[phase_idx] = list(active_in_lanes)
            phase_out_map[phase_idx] = list(active_out_lanes)

        PHASE_LANES[tls_id] = phase_in_map
        PHASE_OUT_LANES[tls_id] = phase_out_map


def get_phase_demand(tls_id, phase):
    """Compute demand as the number of halting vehicles on the phase's incoming lanes."""
    lanes = set(PHASE_LANES[tls_id][phase])
    return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)


def get_phase_pressure(tls_id, phase):
    """Estimate local phase pressure as incoming queue minus downstream spillback."""
    in_lanes = set(PHASE_LANES[tls_id][phase])
    out_lanes = set(PHASE_OUT_LANES[tls_id][phase])
    incoming_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in in_lanes)
    outgoing_queue = sum(
        traci.lane.getLastStepHaltingNumber(lane) for lane in out_lanes
    )
    return incoming_queue - 0.5 * outgoing_queue


MIN_GREEN_MP = (
    25  # MaxPressure minimum hold — close to fixed-time phase durations (24–38 s).
)
# Shorter values cause phase thrashing: yellow clearance overhead eats into green utilization
# and break platoon progression along the corridor.


def select_max_pressure_phase(tls_id, current_phase, elapsed_green):
    """MaxPressure: choose the green phase with highest in/out queue pressure.

    Used as a greedy baseline controller in evaluate.py. Uses MIN_GREEN_MP (not
    MIN_GREEN) to prevent phase thrashing at fine decision granularity.
    """
    green_phases = TLS_CONFIGS[tls_id]["green_phases"]
    best_phase = current_phase
    best_pressure = float("-inf")
    for phase in green_phases:
        pressure = get_phase_pressure(tls_id, phase)
        if pressure > best_pressure:
            best_pressure = pressure
            best_phase = phase
    if best_phase != current_phase and elapsed_green < MIN_GREEN_MP:
        return current_phase
    return best_phase


def get_tls_controlled_lanes(tls_id):
    """Return unique controlled lanes for one TLS."""
    return list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))


def get_phase_position(tls_id, phase):
    """Return the normalized current green-phase position within a TLS cycle."""
    green_phases = TLS_CONFIGS[tls_id]["green_phases"]
    return green_phases.index(phase) / max(len(green_phases) - 1, 1)


def get_neighbor_features(tls_id, metrics_by_tls=None, current_phases=None):
    """Return compact upstream/downstream context for one TLS."""
    features = []
    for neighbor_key in ("prev", "next"):
        neighbor_tls = NEIGHBOR_TLS_IDS[tls_id][neighbor_key]
        if neighbor_tls is None:
            features.extend([0.0, 0.0, 0.0])
            continue

        neighbor_metrics = (
            metrics_by_tls[neighbor_tls]
            if metrics_by_tls is not None and neighbor_tls in metrics_by_tls
            else get_congestion_metrics(neighbor_tls)
        )
        neighbor_queue_pressure = neighbor_metrics["total_queue"] / (
            max(neighbor_metrics["lane_count"], 1) * 12.0
        )
        neighbor_spillback_pressure = neighbor_metrics["total_outgoing_queue"] / (
            max(neighbor_metrics["outgoing_lane_count"], 1) * 12.0
        )
        if current_phases is not None and neighbor_tls in current_phases:
            neighbor_phase_position = get_phase_position(
                neighbor_tls,
                current_phases[neighbor_tls],
            )
        else:
            neighbor_phase_position = 0.0
        features.extend(
            [
                neighbor_queue_pressure,
                neighbor_spillback_pressure,
                neighbor_phase_position,
            ]
        )

    return features


def get_neighbor_ids(tls_id):
    """Return the existing immediate neighbor TLS IDs for one controlled intersection."""
    return [
        neighbor_tls
        for neighbor_tls in (
            NEIGHBOR_TLS_IDS[tls_id]["prev"],
            NEIGHBOR_TLS_IDS[tls_id]["next"],
        )
        if neighbor_tls is not None
    ]


def get_state(
    tls_id, current_phase, elapsed_green, metrics_by_tls=None, current_phases=None
):
    """Build the local state vector for one controlled intersection."""
    green_phases = TLS_CONFIGS[tls_id]["green_phases"]
    phase_pressures = [
        get_phase_pressure(tls_id, phase) / 30.0 for phase in green_phases
    ]
    phase_position = get_phase_position(tls_id, current_phase)
    metrics = (
        metrics_by_tls[tls_id]
        if metrics_by_tls is not None and tls_id in metrics_by_tls
        else get_congestion_metrics(tls_id)
    )
    lane_count = max(metrics["lane_count"], 1)
    queue_pressure = metrics["total_queue"] / (lane_count * 12.0)
    wait_pressure = metrics["total_wait"] / (lane_count * 300.0)
    max_lane_pressure = metrics["max_lane_queue"] / 20.0
    spillback_pressure = metrics["total_outgoing_queue"] / (
        metrics["outgoing_lane_count"] * 12.0
    )
    neighbor_features = get_neighbor_features(
        tls_id,
        metrics_by_tls=metrics_by_tls,
        current_phases=current_phases,
    )
    # Traffic load: vehicles currently in network normalised by typical peak occupancy.
    # Lets the network condition its policy on scenario intensity (hidden variable otherwise).
    traffic_load = traci.vehicle.getIDCount() / 500.0
    return np.array(
        phase_pressures
        + [
            elapsed_green / MAX_GREEN,
            phase_position,
            queue_pressure,
            wait_pressure,
            max_lane_pressure,
            spillback_pressure,
            traffic_load,
        ]
        + neighbor_features,
        dtype=np.float32,
    )


def get_congestion_metrics(tls_id):
    """Compute queue and worst-lane pressure across one TLS's controlled lanes."""
    lanes = get_tls_controlled_lanes(tls_id)
    lane_queues = [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes]
    lane_waits = [traci.lane.getWaitingTime(lane) for lane in lanes]
    outgoing_lanes = sorted(
        {
            lane
            for phase in TLS_CONFIGS[tls_id]["green_phases"]
            for lane in PHASE_OUT_LANES[tls_id].get(phase, [])
        }
    )
    outgoing_queues = [
        traci.lane.getLastStepHaltingNumber(lane) for lane in outgoing_lanes
    ]
    return {
        "total_queue": sum(lane_queues),
        "total_wait": sum(lane_waits),
        "max_lane_queue": max(lane_queues, default=0),
        # Max accumulated wait on any single controlled lane.
        # Detects starvation directly: a lane with one car waiting 60 s reports 60.
        # The reward uses this to overcome the small/noisy signal in low-demand.
        "max_lane_wait": max(lane_waits, default=0.0),
        "lane_count": len(lanes),
        "total_outgoing_queue": sum(outgoing_queues),
        "outgoing_lane_count": max(len(outgoing_lanes), 1),
    }


def get_global_flow_metrics():
    """Return global flow counters shared by the whole network."""
    return {
        "arrived": traci.simulation.getArrivedNumber(),
        "loaded": traci.simulation.getLoadedNumber(),
    }


def build_transition_state(tls_id, current_phase, target_phase):
    """Build a pair-specific yellow transition state for one TLS."""
    current_state = TLS_CONFIGS[tls_id]["phase_states"][current_phase]
    target_state = TLS_CONFIGS[tls_id]["phase_states"][target_phase]
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


def apply_green_phase(tls_id, phase):
    """Apply a green phase by its state string instead of by index."""
    target_state = TLS_CONFIGS[tls_id]["phase_states"][phase]
    traci.trafficlight.setRedYellowGreenState(tls_id, target_state)


def determine_current_green_phase(tls_id):
    """Map SUMO's current phase index to the controller's current green phase."""
    current_phase = traci.trafficlight.getPhase(tls_id)
    green_phases = TLS_CONFIGS[tls_id]["green_phases"]
    if current_phase in green_phases:
        return current_phase

    previous_greens = [phase for phase in green_phases if phase <= current_phase]
    if previous_greens:
        return previous_greens[-1]
    return green_phases[-1]


def decode_action(tls_id, action_id):
    """Map a local action id to either EXTEND or a target green phase."""
    if action_id == ACTION_EXTEND:
        return {"type": "extend", "target_phase": None}

    action_dim = ACTION_DIMS[tls_id]
    if not 0 <= action_id < action_dim:
        raise ValueError(
            f"Unknown action id {action_id} for {tls_id}. Expected range [0, {action_dim - 1}]."
        )

    target_phase = TLS_CONFIGS[tls_id]["green_phases"][action_id - 1]
    return {"type": "switch", "target_phase": target_phase}


def get_valid_action_ids(tls_id, current_phase, elapsed_green):
    """Return action ids that can actually be applied in the current timing state."""
    valid_actions = [ACTION_EXTEND]
    if elapsed_green < MIN_GREEN:
        return valid_actions

    for idx, phase in enumerate(TLS_CONFIGS[tls_id]["green_phases"], start=1):
        if phase != current_phase:
            valid_actions.append(idx)
    return valid_actions


def select_action_from_q_values(q_values, valid_action_ids):
    """Choose the highest-Q valid action while masking invalid actions."""
    valid_indices = torch.LongTensor(valid_action_ids)
    valid_q_values = q_values[valid_indices]
    best_valid_idx = valid_q_values.argmax().item()
    return valid_action_ids[best_valid_idx]


def select_epsilon_greedy_action(policy_net, state, valid_action_ids, epsilon):
    """Sample/exploit only actions that the signal controller can apply."""
    if random.random() < epsilon:
        return random.choice(valid_action_ids)

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state_tensor).squeeze(0)
        return select_action_from_q_values(q_values, valid_action_ids)


def compute_pressure_reward(tls_id, metrics, current_phase, switched):
    """PressLight-style reward with starvation guard and demand-scaled switch cost.

    Two terms that did not exist in the original PressLight formulation are critical
    for arterial corridors with mixed demand profiles:

    1. ``max_wait_term``: dominates whenever any approach is starving (>20 s waited).
       This is what fixes the off-peak collapse: in low demand the queue/pressure
       terms are tiny and noisy, so the agent could not tell extend from switch.
       A single car waiting 60 s now produces a -2.7 contribution that swamps all
       other terms and forces a switch.

    2. ``demand_scale``: in low-demand the constant SWITCH_PENALTY=0.15 is comparable
       to the noise floor of phase_pressure, so the agent thrashes (off-peak Papazyan
       switched 537/918 decisions). Scaling the penalty inversely with local demand
       gives ~1.65 effective penalty in off-peak (strongly anti-thrash) while leaving
       corridor_stress at ~0.225 (the +13% arrivals advantage there is preserved).
    """
    lane_count = max(metrics["lane_count"], 1)
    outgoing_lane_count = metrics["outgoing_lane_count"]

    phase_pressure = (
        sum(
            abs(get_phase_pressure(tls_id, phase))
            for phase in TLS_CONFIGS[tls_id]["green_phases"]
        )
        / lane_count
    )
    queue_pressure = metrics["total_queue"] / lane_count
    wait_pressure = metrics["total_wait"] / (lane_count * 60.0)
    spillback_pressure = metrics["total_outgoing_queue"] / outgoing_lane_count
    served_phase_pressure = (
        max(get_phase_pressure(tls_id, current_phase), 0.0) / lane_count
    )

    # Starvation guard: ramp up sharply once any single lane has waited >20 s.
    # Capped at 60 s of excess (term=2.0): a car waiting 200 s isn't 3x more urgent
    # than 60 s — the agent has already failed. The cap also prevents extreme single
    # states from skewing the demand normalizer ratio in the training loop.
    max_lane_wait = metrics.get("max_lane_wait", 0.0)
    max_wait_excess = min(max(0.0, max_lane_wait - 20.0), 60.0)
    max_wait_term = max_wait_excess / 30.0

    reward = (
        -0.70 * phase_pressure
        - 0.30 * queue_pressure
        - 0.20 * wait_pressure
        - 0.50 * spillback_pressure
        + 0.25 * served_phase_pressure
        - 2.00 * max_wait_term
    )
    if switched:
        # Demand-aware penalty: floor at phase_pressure 0.05 to avoid divide-by-zero.
        demand_scale = 1.0 + 0.5 / max(phase_pressure, 0.05)
        reward -= SWITCH_PENALTY * demand_scale

    return reward


def record_metrics_step(metrics, controlled_lanes):
    """Record queue and waiting-time metrics for evaluation."""
    metrics["steps"] += 1
    metrics["total_queue"] += sum(
        traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes
    )
    metrics["total_lane_wait"] += sum(
        traci.lane.getWaitingTime(lane) for lane in controlled_lanes
    )


def step_simulation(seconds, metrics=None, controlled_lanes=None, render_delay=0.0):
    """Advance the simulation and optionally record evaluation metrics."""
    for _ in range(seconds):
        traci.simulationStep()
        if metrics is not None and controlled_lanes is not None:
            record_metrics_step(metrics, controlled_lanes)
        if render_delay > 0.0:
            time.sleep(render_delay)


def apply_actions(
    action_infos,
    current_phases,
    elapsed_greens,
    metrics=None,
    controlled_lanes=None,
    render_delay=0.0,
):
    """Apply one simultaneous control step across all controlled TLSs."""
    switch_plan = {}

    for tls_id in CONTROLLED_TLS_IDS:
        current_phase = current_phases[tls_id]
        elapsed_green = elapsed_greens[tls_id]
        action_info = action_infos[tls_id]

        should_switch = False
        target_phase = current_phase

        if action_info["type"] == "switch":
            candidate = action_info["target_phase"]
            if candidate != current_phase and elapsed_green >= MIN_GREEN:
                should_switch = True
                target_phase = candidate

        # Prevent greens from overrunning the maximum duration.
        if not should_switch and elapsed_green + EXTEND_STEP >= MAX_GREEN:
            should_switch = True
            target_phase = TLS_CONFIGS[tls_id]["next_green"][current_phase]

        switch_plan[tls_id] = {
            "switched": should_switch,
            "target_phase": target_phase,
        }

    switching_tls_ids = [
        tls_id for tls_id, plan in switch_plan.items() if plan["switched"]
    ]

    if switching_tls_ids:
        for tls_id in switching_tls_ids:
            current_phase = current_phases[tls_id]
            target_phase = switch_plan[tls_id]["target_phase"]
            transition_state = build_transition_state(
                tls_id, current_phase, target_phase
            )
            if transition_state != traci.trafficlight.getRedYellowGreenState(tls_id):
                traci.trafficlight.setRedYellowGreenState(tls_id, transition_state)

        step_simulation(
            YELLOW_DURATION,
            metrics=metrics,
            controlled_lanes=controlled_lanes,
            render_delay=render_delay,
        )

        for tls_id in CONTROLLED_TLS_IDS:
            if switch_plan[tls_id]["switched"]:
                target_phase = switch_plan[tls_id]["target_phase"]
                apply_green_phase(tls_id, target_phase)
                current_phases[tls_id] = target_phase
                elapsed_greens[tls_id] = 0
            else:
                elapsed_greens[tls_id] += YELLOW_DURATION

    step_simulation(
        EXTEND_STEP,
        metrics=metrics,
        controlled_lanes=controlled_lanes,
        render_delay=render_delay,
    )

    for tls_id in CONTROLLED_TLS_IDS:
        elapsed_greens[tls_id] += EXTEND_STEP

    return current_phases, elapsed_greens, switch_plan


def build_agents():
    """Create one local DQN agent per controlled intersection."""
    agents = {}
    for tls_id in CONTROLLED_TLS_IDS:
        state_dim = STATE_DIMS[tls_id]
        action_dim = ACTION_DIMS[tls_id]
        policy_net = DQN(state_dim, action_dim)
        target_net = DQN(state_dim, action_dim)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        agents[tls_id] = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "policy_net": policy_net,
            "target_net": target_net,
            "optimizer": optim.Adam(policy_net.parameters(), lr=LR),
            "memory": ReplayBuffer(REPLAY_BUFFER_SIZE),
        }

    return agents


def save_model_bundle(model_path, agents):
    """Save all local policies and metadata into one checkpoint bundle."""
    payload = {
        "controlled_tls_ids": CONTROLLED_TLS_IDS,
        "sumo_config": SUMO_CONFIG,
        "agents": {
            tls_id: {
                "state_dict": agents[tls_id]["policy_net"].state_dict(),
                "state_dim": agents[tls_id]["state_dim"],
                "action_dim": agents[tls_id]["action_dim"],
                "green_phases": TLS_CONFIGS[tls_id]["green_phases"],
            }
            for tls_id in CONTROLLED_TLS_IDS
        },
    }
    torch.save(payload, model_path)


def load_model_bundle(model_path, device="cpu"):
    """Load the saved local DQN policies."""
    payload = torch.load(model_path, map_location=device, weights_only=False)
    if "agents" not in payload:
        raise RuntimeError(
            "Saved model is incompatible with the current multi-intersection controller. "
            "Retrain with train.py and try again."
        )

    policy_nets = {}
    for tls_id in CONTROLLED_TLS_IDS:
        if tls_id not in payload["agents"]:
            raise RuntimeError(f"Saved model is missing policy weights for {tls_id}.")
        spec = payload["agents"][tls_id]
        policy_net = DQN(spec["state_dim"], spec["action_dim"])
        policy_net.load_state_dict(spec["state_dict"])
        policy_net.eval()
        policy_nets[tls_id] = policy_net

    return policy_nets


def train(
    episodes=100,
    decisions_per_episode=720,
    model_path=MODEL_PATH,
    checkpoint_every=None,
    checkpoint_dir="runs/current/checkpoints",
    route_file=None,
    training_scenarios=None,
    training_seeds=None,
    training_steps=3600,
    training_route_dir="runs/current/training_routes",
    log_csv="training_log.csv",
):
    """Train one local DQN controller per selected intersection."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    if log_csv:
        _log_path = Path(log_csv)
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        with _log_path.open("w", encoding="utf-8") as _f:
            _f.write(
                "episode,total_reward,avg_reward_per_agent_decision,"
                "avg_loss,epsilon,scenario\n"
            )

    training_routes = None
    if route_file is None:
        training_routes = prepare_training_route_files(
            scenarios=training_scenarios or DEFAULT_TRAINING_SCENARIOS,
            seeds=training_seeds or DEFAULT_TRAINING_SEEDS,
            steps=training_steps,
            output_dir=training_route_dir,
        )

    agents = build_agents()
    batch_size = 64
    gamma = 0.99
    target_update_freq = TARGET_UPDATE_FREQ
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = max(1, int(EPSILON_DECAY_RATIO * episodes))
    loss_fn = nn.MSELoss()

    checkpoint_path = None
    if checkpoint_every:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Scenario-balanced episode schedule: round-robin over scenarios first, then
    # cycle seeds within each scenario. This guarantees equal exposure across
    # scenarios over any window of (n_scenarios) episodes — far better than
    # random.choice which can produce 4-of-10 same-scenario streaks that skew
    # the replay buffer toward whichever demand profile happened to win the
    # coin flip lately.
    scenario_schedule = None
    if training_routes is not None:
        seen_scenarios = []
        for r in training_routes:
            if r["scenario"] not in seen_scenarios:
                seen_scenarios.append(r["scenario"])
        routes_by_scenario = {
            s: [r for r in training_routes if r["scenario"] == s]
            for s in seen_scenarios
        }
        seed_cursor = {s: 0 for s in seen_scenarios}
        scenario_schedule = []
        for ep in range(episodes):
            scenario = seen_scenarios[ep % len(seen_scenarios)]
            seed_idx = seed_cursor[scenario] % len(routes_by_scenario[scenario])
            seed_cursor[scenario] += 1
            scenario_schedule.append(routes_by_scenario[scenario][seed_idx])

    # Per-scenario rolling avg reward (last 10 episodes per scenario). Lets you
    # confirm the demand normalizer actually equalized magnitudes across scenarios
    # — if the printout still shows a wide spread, the normalizer is mis-tuned.
    scenario_recent_rewards = defaultdict(lambda: deque(maxlen=10))

    for episode in range(episodes):
        episode_route = route_file
        episode_label = "fixed route"
        episode_scenario = "fixed"
        if scenario_schedule is not None:
            selected_route = scenario_schedule[episode]
            episode_route = selected_route["route_file"]
            episode_label = (
                f"{selected_route['scenario']} seed {selected_route['seed']}"
            )
            episode_scenario = selected_route["scenario"]

        config_path, temp_config_path = create_config_with_route_override(episode_route)
        sumo_cmd = build_sumo_cmd(config_path)

        try:
            start_traci(sumo_cmd)
            try:
                init_phase_lanes()
                current_phases = {
                    tls_id: determine_current_green_phase(tls_id)
                    for tls_id in CONTROLLED_TLS_IDS
                }
                elapsed_greens = {tls_id: 0 for tls_id in CONTROLLED_TLS_IDS}

                step_for_seconds(20)

                episode_losses = {tls_id: [] for tls_id in CONTROLLED_TLS_IDS}
                episode_reward = 0.0
                episode_transitions = 0
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
                    0, (epsilon_decay - episode) / epsilon_decay
                )

                for step in range(decisions_per_episode):
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        break

                    prev_local_metrics = {
                        tls_id: get_congestion_metrics(tls_id)
                        for tls_id in CONTROLLED_TLS_IDS
                    }
                    states = {
                        tls_id: get_state(
                            tls_id,
                            current_phases[tls_id],
                            elapsed_greens[tls_id],
                            metrics_by_tls=prev_local_metrics,
                            current_phases=current_phases,
                        )
                        for tls_id in CONTROLLED_TLS_IDS
                    }
                    prev_global_metrics = get_global_flow_metrics()

                    action_ids = {}
                    action_infos = {}
                    for tls_id in CONTROLLED_TLS_IDS:
                        valid_action_ids = get_valid_action_ids(
                            tls_id,
                            current_phases[tls_id],
                            elapsed_greens[tls_id],
                        )
                        action_id = select_epsilon_greedy_action(
                            agents[tls_id]["policy_net"],
                            states[tls_id],
                            valid_action_ids,
                            epsilon,
                        )
                        action_ids[tls_id] = action_id
                        action_infos[tls_id] = decode_action(tls_id, action_id)

                    current_phases, elapsed_greens, switch_plan = apply_actions(
                        action_infos,
                        current_phases,
                        elapsed_greens,
                    )

                    # Resolve what was actually applied (blocked switches become EXTEND)
                    applied_action_ids = {}
                    for _tid in CONTROLLED_TLS_IDS:
                        if switch_plan[_tid]["switched"]:
                            _target = switch_plan[_tid]["target_phase"]
                            _idx = TLS_CONFIGS[_tid]["green_phases"].index(_target)
                            applied_action_ids[_tid] = _idx + 1
                        else:
                            applied_action_ids[_tid] = ACTION_EXTEND

                    current_local_metrics = {
                        tls_id: get_congestion_metrics(tls_id)
                        for tls_id in CONTROLLED_TLS_IDS
                    }
                    current_global_metrics = get_global_flow_metrics()

                    throughput_delta = (
                        current_global_metrics["arrived"]
                        - prev_global_metrics["arrived"]
                    )
                    throughput_bonus = 0.10 * throughput_delta / len(CONTROLLED_TLS_IDS)

                    # Demand-aware reward normalizer.
                    #
                    # Without this, raw rewards differ ~9x across scenarios
                    # (off_peak ~-0.16 vs corridor_stress ~-1.37 avg per agent
                    # decision), so corridor_stress transitions dominate TD-error
                    # gradients and the shared DQN overfits to heavy-traffic
                    # behavior. We divide by max(n_vehicles/50, 1.0):
                    #   - off_peak (~30 vehicles): divisor 1.0, no change
                    #   - rush_hour (~150): divisor 3.0
                    #   - corridor_stress (~300): divisor 6.0
                    # Equalizes per-step gradient magnitude across scenarios while
                    # preserving relative action ordering within each scenario, so
                    # the optimal policy is unchanged in MDP terms.
                    n_vehicles_now = max(traci.vehicle.getIDCount(), 1)
                    demand_normalizer = max(n_vehicles_now / 50.0, 1.0)

                    done = (
                        step == (decisions_per_episode - 1)
                        or traci.simulation.getMinExpectedNumber() <= 0
                    )
                    for tls_id in CONTROLLED_TLS_IDS:
                        reward = compute_pressure_reward(
                            tls_id,
                            current_local_metrics[tls_id],
                            current_phases[tls_id],
                            switch_plan[tls_id]["switched"],
                        )
                        reward += throughput_bonus
                        reward -= 0.02 * max(
                            0, elapsed_greens[tls_id] - STARVATION_GREEN
                        )
                        reward /= REWARD_SCALE * demand_normalizer
                        next_state = get_state(
                            tls_id,
                            current_phases[tls_id],
                            elapsed_greens[tls_id],
                            metrics_by_tls=current_local_metrics,
                            current_phases=current_phases,
                        )
                        next_valid_action_ids = get_valid_action_ids(
                            tls_id,
                            current_phases[tls_id],
                            elapsed_greens[tls_id],
                        )
                        agents[tls_id]["memory"].push(
                            states[tls_id],
                            applied_action_ids[tls_id],
                            reward,
                            next_state,
                            done,
                            next_valid_action_ids,
                        )
                        episode_reward += reward
                        episode_transitions += 1

                    for tls_id in CONTROLLED_TLS_IDS:
                        memory = agents[tls_id]["memory"]
                        if len(memory) < batch_size:
                            continue

                        (
                            states_batch,
                            actions_batch,
                            rewards_batch,
                            next_states_batch,
                            dones_batch,
                            next_valid_actions_batch,
                        ) = memory.sample(batch_size)

                        states_tensor = torch.FloatTensor(states_batch)
                        actions_tensor = torch.LongTensor(actions_batch).unsqueeze(1)
                        rewards_tensor = torch.FloatTensor(rewards_batch).unsqueeze(1)
                        next_states_tensor = torch.FloatTensor(next_states_batch)
                        dones_tensor = torch.FloatTensor(dones_batch).unsqueeze(1)

                        policy_net = agents[tls_id]["policy_net"]
                        target_net = agents[tls_id]["target_net"]
                        optimizer = agents[tls_id]["optimizer"]

                        q_values = policy_net(states_tensor).gather(1, actions_tensor)
                        with torch.no_grad():
                            next_policy_q = policy_net(next_states_tensor)
                            invalid_mask = torch.full_like(next_policy_q, -1e9)
                            for row_idx, valid_actions in enumerate(
                                next_valid_actions_batch
                            ):
                                invalid_mask[row_idx, valid_actions] = 0.0
                            next_actions = (next_policy_q + invalid_mask).argmax(
                                1, keepdim=True
                            )
                            next_q_values = target_net(next_states_tensor).gather(
                                1, next_actions
                            )
                            target_q_values = rewards_tensor + (
                                gamma * next_q_values * (1 - dones_tensor)
                            )

                        loss = loss_fn(q_values, target_q_values)
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                        optimizer.step()
                        episode_losses[tls_id].append(loss.item())
            finally:
                traci.close()
        finally:
            if temp_config_path:
                os.unlink(temp_config_path)

        if episode % target_update_freq == 0:
            for tls_id in CONTROLLED_TLS_IDS:
                agents[tls_id]["target_net"].load_state_dict(
                    agents[tls_id]["policy_net"].state_dict()
                )

        all_losses = [v for losses in episode_losses.values() for v in losses]
        avg_loss_val = sum(all_losses) / len(all_losses) if all_losses else float("nan")
        avg_loss_str = f"{avg_loss_val:.4f}" if all_losses else "n/a"
        avg_reward = episode_reward / max(episode_transitions, 1)
        scenario_recent_rewards[episode_scenario].append(avg_reward)
        print(
            f"Episode {episode + 1}/{episodes}: reward = {episode_reward:.2f} | "
            f"avg_reward = {avg_reward:.4f} | loss = {avg_loss_str} | "
            f"eps = {epsilon:.3f} | route = {episode_label}"
        )
        if log_csv:
            with Path(log_csv).open("a", encoding="utf-8") as _f:
                _f.write(
                    f"{episode + 1},{episode_reward:.4f},"
                    f"{avg_reward:.6f},"
                    f"{'nan' if not all_losses else f'{avg_loss_val:.6f}'},"
                    f"{epsilon:.4f},{episode_label}\n"
                )

        if (episode + 1) % 10 == 0 and scenario_recent_rewards:
            print(
                "  --- Per-scenario rolling avg reward (last 10 episodes/scenario) ---"
            )
            for sc in sorted(scenario_recent_rewards):
                rewards_dq = scenario_recent_rewards[sc]
                if rewards_dq:
                    mean_r = sum(rewards_dq) / len(rewards_dq)
                    spread = max(rewards_dq) - min(rewards_dq)
                    print(
                        f"    {sc:18s} avg={mean_r:+.4f}  spread={spread:.4f}  n={len(rewards_dq)}"
                    )

        if checkpoint_path and (episode + 1) % checkpoint_every == 0:
            checkpoint_file = checkpoint_path / f"checkpoint_ep{episode + 1:03d}.pth"
            save_model_bundle(checkpoint_file, agents)
            print(f"Saved checkpoint to {checkpoint_file}")

    save_model_bundle(model_path, agents)
    print(f"Training complete! Model saved to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train local DQN traffic light controllers for the Komitas corridor."
    )
    parser.add_argument(
        "--episodes", type=int, default=400, help="Number of training episodes."
    )
    parser.add_argument(
        "--decisions-per-episode",
        type=int,
        default=720,
        help="Number of control decisions per training episode.",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to save the final trained model bundle.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save a checkpoint every N episodes. Set to 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="runs/current/checkpoints",
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
        default="runs/current/training_routes",
        help="Directory used for generated training route files.",
    )
    parser.add_argument(
        "--log-csv",
        default="training_log.csv",
        help="Path to write the per-episode training log CSV.",
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
        log_csv=args.log_csv,
    )

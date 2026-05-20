"""Evaluate the legacy single-intersection DQN controller.

This script preserves the older Komitas-Vagharshyan single-intersection
evaluation path. The active project code is multi-intersection, but the paper
keeps the single-intersection result as historical reference evidence.
"""

import argparse
import json
import os
import socket
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import traci


ROOT = Path(__file__).resolve().parents[1]
SUMO_CONFIG = (
    ROOT
    / "data"
    / "raw_data"
    / "sumo_data"
    / "legacy_single_intersection"
    / "komitas-vagharshyan.sumocfg"
)
DEFAULT_MODEL = ROOT / "code" / "models" / "dqn_single.pth"
DEFAULT_ROUTE = (
    ROOT / "data" / "processed_data" / "single_intersection_evening_rush_seed42.rou.xml"
)

TLS_ID = "cluster_11668441165_11668441166_11668441167_2912634528_#8more"
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

EXTEND_STEP = 3
YELLOW_DURATION = 3
MIN_GREEN = 5
MAX_GREEN = 90

PHASE_LANES = {}


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


def create_config_with_route_override(config_path, route_file=None):
    """Create a temporary SUMO config with an optional absolute route override."""
    config_path = Path(config_path)
    if route_file is None:
        return str(config_path), None

    tree = ET.parse(config_path)
    root = tree.getroot()
    config_dir = config_path.resolve().parent
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


def build_eval_sumo_cmd(config_path, tripinfo_output):
    sumo_binary = os.environ.get("SUMO_EVAL_BINARY", "sumo")
    return [
        sumo_binary,
        "-c",
        str(config_path),
        "--no-warnings",
        "--no-step-log",
        "--quit-on-end",
        "--tripinfo-output",
        tripinfo_output,
    ]


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
    """Build the phase-to-incoming-lane mapping from the loaded SUMO network."""
    PHASE_LANES.clear()
    logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
    controlled_links = traci.trafficlight.getControlledLinks(TLS_ID)

    for phase_idx in GREEN_PHASES:
        state = logic.phases[phase_idx].state
        active_in_lanes = set()
        for link_idx, signal_char in enumerate(state):
            if signal_char in ("G", "g", "s") and link_idx < len(controlled_links):
                for connection in controlled_links[link_idx]:
                    active_in_lanes.add(connection[0])
        PHASE_LANES[phase_idx] = list(active_in_lanes)


def get_phase_demand(phase):
    lanes = set(PHASE_LANES[phase])
    return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)


def get_state(current_phase, elapsed_green):
    phase_demands = [get_phase_demand(phase) / 50.0 for phase in GREEN_PHASES]
    phase_position = GREEN_PHASES.index(current_phase) / (len(GREEN_PHASES) - 1)
    return np.array(
        phase_demands + [elapsed_green / MAX_GREEN, phase_position],
        dtype=np.float32,
    )


def build_transition_state(current_phase, target_phase):
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
    logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
    traci.trafficlight.setRedYellowGreenState(TLS_ID, logic.phases[phase].state)


def decode_action(action_id):
    if action_id == ACTION_EXTEND:
        return {"type": "extend", "target_phase": None}
    if not 0 <= action_id < ACTION_DIM:
        raise ValueError(f"Unknown action id {action_id}.")
    return {"type": "switch", "target_phase": GREEN_PHASES[action_id - 1]}


def parse_tripinfo(path):
    root = ET.parse(path).getroot()
    trips = root.findall("tripinfo")
    if not trips:
        return {
            "arrived_vehicles": 0,
            "avg_trip_duration": 0.0,
            "avg_waiting_time": 0.0,
            "avg_time_loss": 0.0,
            "max_waiting_time": 0.0,
        }

    durations = [float(trip.attrib["duration"]) for trip in trips]
    waiting_times = [float(trip.attrib.get("waitingTime", 0.0)) for trip in trips]
    time_losses = [float(trip.attrib.get("timeLoss", 0.0)) for trip in trips]
    return {
        "arrived_vehicles": len(trips),
        "avg_trip_duration": sum(durations) / len(durations),
        "avg_waiting_time": sum(waiting_times) / len(waiting_times),
        "avg_time_loss": sum(time_losses) / len(time_losses),
        "max_waiting_time": max(waiting_times),
    }


def init_metrics():
    return {"steps": 0, "total_queue": 0.0, "total_lane_wait": 0.0}


def record_step_metrics(metrics, controlled_lanes):
    metrics["steps"] += 1
    metrics["total_queue"] += sum(
        traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes
    )
    metrics["total_lane_wait"] += sum(
        traci.lane.getWaitingTime(lane) for lane in controlled_lanes
    )


def step_with_metrics(seconds, metrics, controlled_lanes):
    for _ in range(seconds):
        traci.simulationStep()
        record_step_metrics(metrics, controlled_lanes)


def finalize_metrics(metrics, tripinfo_metrics):
    steps = max(metrics["steps"], 1)
    return {
        "steps": metrics["steps"],
        "arrived_vehicles": tripinfo_metrics["arrived_vehicles"],
        "avg_queue_per_step": metrics["total_queue"] / steps,
        "avg_lane_wait_per_step": metrics["total_lane_wait"] / steps,
        "avg_trip_duration": tripinfo_metrics["avg_trip_duration"],
        "avg_waiting_time": tripinfo_metrics["avg_waiting_time"],
        "avg_time_loss": tripinfo_metrics["avg_time_loss"],
        "max_waiting_time": tripinfo_metrics["max_waiting_time"],
    }


def run_default_controller(config_path=SUMO_CONFIG, route_file=DEFAULT_ROUTE):
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tripinfo_path = tmp.name

    metrics = init_metrics()
    config_path, temp_config_path = create_config_with_route_override(
        config_path, route_file
    )
    start_traci(build_eval_sumo_cmd(config_path, tripinfo_path))
    try:
        controlled_lanes = list(set(traci.trafficlight.getControlledLanes(TLS_ID)))
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            record_step_metrics(metrics, controlled_lanes)
    finally:
        traci.close()
        if temp_config_path:
            os.unlink(temp_config_path)

    tripinfo_metrics = parse_tripinfo(tripinfo_path)
    os.unlink(tripinfo_path)
    return finalize_metrics(metrics, tripinfo_metrics)


def load_policy(model_path):
    policy_net = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    policy_net.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy_net.eval()
    return policy_net


def run_rl_controller(
    model_path=DEFAULT_MODEL,
    config_path=SUMO_CONFIG,
    route_file=DEFAULT_ROUTE,
):
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tripinfo_path = tmp.name

    metrics = init_metrics()
    policy_net = load_policy(model_path)
    config_path, temp_config_path = create_config_with_route_override(
        config_path, route_file
    )
    start_traci(build_eval_sumo_cmd(config_path, tripinfo_path))
    try:
        init_phase_lanes()
        controlled_lanes = list(set(traci.trafficlight.getControlledLanes(TLS_ID)))
        current_phase = traci.trafficlight.getPhase(TLS_ID)
        elapsed_green = 0

        step_with_metrics(20, metrics, controlled_lanes)

        while traci.simulation.getMinExpectedNumber() > 0:
            state = get_state(current_phase, elapsed_green)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = policy_net(state_tensor).argmax().item()

            action_info = decode_action(action)
            if action_info["type"] == "extend":
                step_with_metrics(EXTEND_STEP, metrics, controlled_lanes)
                elapsed_green += EXTEND_STEP
                if elapsed_green >= MAX_GREEN:
                    target_phase = PHASE_TRANSITIONS[current_phase]["next_green"]
                    current_phase = switch_to_phase_with_metrics(
                        current_phase, target_phase, metrics, controlled_lanes
                    )
                    elapsed_green = 0
            else:
                target_phase = action_info["target_phase"]
                if target_phase == current_phase or elapsed_green < MIN_GREEN:
                    step_with_metrics(EXTEND_STEP, metrics, controlled_lanes)
                    elapsed_green += EXTEND_STEP
                else:
                    current_phase = switch_to_phase_with_metrics(
                        current_phase, target_phase, metrics, controlled_lanes
                    )
                    elapsed_green = 0
                    step_with_metrics(EXTEND_STEP, metrics, controlled_lanes)
    finally:
        traci.close()
        if temp_config_path:
            os.unlink(temp_config_path)

    tripinfo_metrics = parse_tripinfo(tripinfo_path)
    os.unlink(tripinfo_path)
    return finalize_metrics(metrics, tripinfo_metrics)


def switch_to_phase_with_metrics(
    current_phase, target_phase, metrics, controlled_lanes
):
    transition_state = build_transition_state(current_phase, target_phase)
    if transition_state != traci.trafficlight.getRedYellowGreenState(TLS_ID):
        traci.trafficlight.setRedYellowGreenState(TLS_ID, transition_state)
        step_with_metrics(YELLOW_DURATION, metrics, controlled_lanes)
    apply_green_phase(target_phase)
    return target_phase


def evaluate_pair(
    model_path=DEFAULT_MODEL, config_path=SUMO_CONFIG, route_file=DEFAULT_ROUTE
):
    return {
        "baseline": run_default_controller(
            config_path=config_path, route_file=route_file
        ),
        "rl": run_rl_controller(
            model_path=model_path, config_path=config_path, route_file=route_file
        ),
    }


def print_comparison(results):
    headers = ("Metric", "Fixed-Time", "RL")
    rows = [
        ("Steps", f"{results['baseline']['steps']}", f"{results['rl']['steps']}"),
        (
            "Vehicles Arrived",
            f"{results['baseline']['arrived_vehicles']}",
            f"{results['rl']['arrived_vehicles']}",
        ),
        (
            "Avg Queue / Step",
            f"{results['baseline']['avg_queue_per_step']:.2f}",
            f"{results['rl']['avg_queue_per_step']:.2f}",
        ),
        (
            "Avg Lane Wait / Step",
            f"{results['baseline']['avg_lane_wait_per_step']:.2f}",
            f"{results['rl']['avg_lane_wait_per_step']:.2f}",
        ),
        (
            "Avg Trip Duration",
            f"{results['baseline']['avg_trip_duration']:.2f}",
            f"{results['rl']['avg_trip_duration']:.2f}",
        ),
        (
            "Avg Waiting Time",
            f"{results['baseline']['avg_waiting_time']:.2f}",
            f"{results['rl']['avg_waiting_time']:.2f}",
        ),
        (
            "Avg Time Loss",
            f"{results['baseline']['avg_time_loss']:.2f}",
            f"{results['rl']['avg_time_loss']:.2f}",
        ),
        (
            "Max Waiting Time",
            f"{results['baseline']['max_waiting_time']:.2f}",
            f"{results['rl']['max_waiting_time']:.2f}",
        ),
    ]

    widths = [
        max(len(headers[0]), *(len(row[0]) for row in rows)),
        max(len(headers[1]), *(len(row[1]) for row in rows)),
        max(len(headers[2]), *(len(row[2]) for row in rows)),
    ]
    print(
        f"{headers[0]:<{widths[0]}}  "
        f"{headers[1]:>{widths[1]}}  "
        f"{headers[2]:>{widths[2]}}"
    )
    print(f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}")
    for metric, baseline, rl in rows:
        print(f"{metric:<{widths[0]}}  {baseline:>{widths[1]}}  {rl:>{widths[2]}}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the legacy single-intersection DQN controller."
    )
    parser.add_argument(
        "--mode",
        choices=("both", "baseline", "rl"),
        default="both",
        help="Which controller runs to execute.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL),
        help="Path to the single-intersection model state dict.",
    )
    parser.add_argument(
        "--config",
        default=str(SUMO_CONFIG),
        help="Legacy single-intersection SUMO config.",
    )
    parser.add_argument(
        "--route-file",
        default=str(DEFAULT_ROUTE),
        help="Route file used for both fixed-time and RL evaluation.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path where raw evaluation metrics are written as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}
    if args.mode in ("both", "baseline"):
        results["baseline"] = run_default_controller(
            config_path=args.config, route_file=args.route_file
        )
    if args.mode in ("both", "rl"):
        results["rl"] = run_rl_controller(
            model_path=args.model_path,
            config_path=args.config,
            route_file=args.route_file,
        )

    if args.mode == "both":
        print_comparison(results)
    else:
        controller = "baseline" if args.mode == "baseline" else "rl"
        for key, value in results[controller].items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

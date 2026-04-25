import argparse
import os
import tempfile
import xml.etree.ElementTree as ET

import torch
import traci

from train import (
    ACTION_DIM,
    apply_green_phase,
    DQN,
    build_transition_state,
    decode_action,
    EXTEND_STEP,
    PHASE_TRANSITIONS,
    MAX_GREEN,
    MIN_GREEN,
    MODEL_PATH,
    STATE_DIM,
    TLS_ID,
    YELLOW_DURATION,
    create_config_with_route_override,
    get_state,
    init_phase_lanes,
)


def build_eval_sumo_cmd(config_path, tripinfo_output):
    """Build a headless SUMO command for evaluation."""
    sumo_binary = os.environ.get("SUMO_EVAL_BINARY", "sumo")
    return [
        sumo_binary,
        "-c",
        config_path,
        "--no-warnings",
        "--quit-on-end",
        "--tripinfo-output",
        tripinfo_output,
    ]


def parse_tripinfo(path):
    """Parse SUMO tripinfo output into aggregate metrics."""
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
    return {
        "steps": 0,
        "total_queue": 0.0,
        "total_lane_wait": 0.0,
    }


def record_step_metrics(metrics, controlled_lanes):
    """Update per-step queue and waiting-time aggregates."""
    metrics["steps"] += 1
    metrics["total_queue"] += sum(
        traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes
    )
    metrics["total_lane_wait"] += sum(
        traci.lane.getWaitingTime(lane) for lane in controlled_lanes
    )


def step_with_metrics(seconds, metrics, controlled_lanes):
    """Advance the simulation and record metrics at each second."""
    for _ in range(seconds):
        traci.simulationStep()
        record_step_metrics(metrics, controlled_lanes)


def switch_phase_with_metrics(current_phase, metrics, controlled_lanes):
    """Transition to the next green phase using a synthesized yellow phase."""
    next_phase = PHASE_TRANSITIONS[current_phase]["next_green"]
    transition_state = build_transition_state(current_phase, next_phase)
    if transition_state != traci.trafficlight.getRedYellowGreenState(TLS_ID):
        traci.trafficlight.setRedYellowGreenState(TLS_ID, transition_state)
        step_with_metrics(YELLOW_DURATION, metrics, controlled_lanes)

    apply_green_phase(next_phase)
    return next_phase


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


def run_default_controller(route_file=None):
    """Run the scenario with the built-in SUMO traffic light logic."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tripinfo_path = tmp.name

    metrics = init_metrics()
    config_path, temp_config_path = create_config_with_route_override(route_file)
    traci.start(build_eval_sumo_cmd(config_path, tripinfo_path))
    try:
        try:
            controlled_lanes = list(set(traci.trafficlight.getControlledLanes(TLS_ID)))
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                record_step_metrics(metrics, controlled_lanes)
        finally:
            traci.close()
    finally:
        if temp_config_path:
            os.unlink(temp_config_path)

    tripinfo_metrics = parse_tripinfo(tripinfo_path)
    os.unlink(tripinfo_path)
    return finalize_metrics(metrics, tripinfo_metrics)


def load_policy(model_path):
    """Load the trained DQN policy from disk."""
    policy_net = DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location="cpu"))
    except RuntimeError as exc:
        raise RuntimeError(
            "Saved model is incompatible with the current controller architecture. "
            "Retrain with train.py before running evaluation."
        ) from exc
    policy_net.eval()
    return policy_net


def run_rl_controller(model_path=MODEL_PATH, route_file=None):
    """Run the scenario using the trained RL controller."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tripinfo_path = tmp.name

    metrics = init_metrics()
    policy_net = load_policy(model_path)
    config_path, temp_config_path = create_config_with_route_override(route_file)

    traci.start(build_eval_sumo_cmd(config_path, tripinfo_path))
    try:
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
                        current_phase = switch_phase_with_metrics(
                            current_phase, metrics, controlled_lanes
                        )
                        elapsed_green = 0
                else:
                    target_phase = action_info["target_phase"]
                    if target_phase == current_phase:
                        step_with_metrics(EXTEND_STEP, metrics, controlled_lanes)
                        elapsed_green += EXTEND_STEP
                    elif elapsed_green < MIN_GREEN:
                        step_with_metrics(EXTEND_STEP, metrics, controlled_lanes)
                        elapsed_green += EXTEND_STEP
                    else:
                        transition_state = build_transition_state(
                            current_phase, target_phase
                        )
                        if (
                            transition_state
                            != traci.trafficlight.getRedYellowGreenState(TLS_ID)
                        ):
                            traci.trafficlight.setRedYellowGreenState(
                                TLS_ID, transition_state
                            )
                            step_with_metrics(
                                YELLOW_DURATION, metrics, controlled_lanes
                            )
                        apply_green_phase(target_phase)
                        current_phase = target_phase
                        elapsed_green = 0
                        step_with_metrics(EXTEND_STEP, metrics, controlled_lanes)
        finally:
            traci.close()
    finally:
        if temp_config_path:
            os.unlink(temp_config_path)

    tripinfo_metrics = parse_tripinfo(tripinfo_path)
    os.unlink(tripinfo_path)
    return finalize_metrics(metrics, tripinfo_metrics)


def evaluate_pair(model_path=MODEL_PATH, route_file=None):
    """Run baseline and RL evaluation on the same scenario."""
    return {
        "baseline": run_default_controller(route_file=route_file),
        "rl": run_rl_controller(model_path=model_path, route_file=route_file),
    }


def print_comparison(results):
    """Print metrics in a simple side-by-side table."""
    headers = ("Metric", "Baseline", "RL")
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
        f"{headers[0]:<{widths[0]}}  {headers[1]:>{widths[1]}}  {headers[2]:>{widths[2]}}"
    )
    print(f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}")
    for metric, baseline, rl in rows:
        print(f"{metric:<{widths[0]}}  {baseline:>{widths[1]}}  {rl:>{widths[2]}}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the default SUMO controller against the trained RL controller."
    )
    parser.add_argument(
        "--mode",
        choices=("both", "baseline", "rl"),
        default="both",
        help="Which controller runs to execute.",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to the trained model used for RL evaluation.",
    )
    parser.add_argument(
        "--route-file",
        default=None,
        help="Optional route file override used for both baseline and RL evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}

    if args.mode in ("both", "baseline"):
        results["baseline"] = run_default_controller(route_file=args.route_file)
    if args.mode in ("both", "rl"):
        results["rl"] = run_rl_controller(
            model_path=args.model_path, route_file=args.route_file
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


if __name__ == "__main__":
    main()

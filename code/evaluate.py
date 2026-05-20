import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter

import torch
import traci

from train import (
    CONTROLLED_TLS_IDS,
    MODEL_PATH,
    apply_actions,
    create_config_with_route_override,
    decode_action,
    determine_current_green_phase,
    get_congestion_metrics,
    get_state,
    get_valid_action_ids,
    init_phase_lanes,
    load_model_bundle,
    select_action_from_q_values,
    select_max_pressure_phase,
    start_traci,
)


def build_eval_sumo_cmd(config_path, tripinfo_output, end_time=7200):
    """Build a headless SUMO command for evaluation."""
    sumo_binary = os.environ.get("SUMO_EVAL_BINARY", "sumo")
    return [
        sumo_binary,
        "-c",
        config_path,
        "--no-warnings",
        "--no-step-log",
        "--quit-on-end",
        "--end",
        str(end_time),
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


def get_all_controlled_lanes():
    """Return the unique union of lanes controlled by the selected TLS set."""
    lanes = []
    for tls_id in CONTROLLED_TLS_IDS:
        lanes.extend(traci.trafficlight.getControlledLanes(tls_id))
    return list(dict.fromkeys(lanes))


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


def run_default_controller(route_file=None, end_time=7200):
    """Run the scenario with the built-in SUMO traffic light logic."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tripinfo_path = tmp.name

    metrics = init_metrics()
    config_path, temp_config_path = create_config_with_route_override(route_file)
    start_traci(build_eval_sumo_cmd(config_path, tripinfo_path, end_time=end_time))
    try:
        try:
            controlled_lanes = get_all_controlled_lanes()
            while (
                traci.simulation.getMinExpectedNumber() > 0
                and traci.simulation.getTime() < end_time
            ):
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


def run_rl_controller(model_path=MODEL_PATH, route_file=None, end_time=7200):
    """Run the scenario using the trained multi-intersection RL controller."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tripinfo_path = tmp.name

    metrics = init_metrics()
    action_counts = {tls_id: Counter() for tls_id in CONTROLLED_TLS_IDS}
    switch_counts = {tls_id: 0 for tls_id in CONTROLLED_TLS_IDS}
    blocked_switch_counts = {tls_id: 0 for tls_id in CONTROLLED_TLS_IDS}
    policy_nets = load_model_bundle(model_path)
    config_path, temp_config_path = create_config_with_route_override(route_file)

    start_traci(build_eval_sumo_cmd(config_path, tripinfo_path, end_time=end_time))
    try:
        try:
            init_phase_lanes()
            controlled_lanes = get_all_controlled_lanes()
            current_phases = {
                tls_id: determine_current_green_phase(tls_id)
                for tls_id in CONTROLLED_TLS_IDS
            }
            elapsed_greens = {tls_id: 0 for tls_id in CONTROLLED_TLS_IDS}

            step_with_metrics(20, metrics, controlled_lanes)

            while (
                traci.simulation.getMinExpectedNumber() > 0
                and traci.simulation.getTime() < end_time
            ):
                local_metrics = {
                    tls_id: get_congestion_metrics(tls_id)
                    for tls_id in CONTROLLED_TLS_IDS
                }
                action_infos = {}
                for tls_id in CONTROLLED_TLS_IDS:
                    state = get_state(
                        tls_id,
                        current_phases[tls_id],
                        elapsed_greens[tls_id],
                        metrics_by_tls=local_metrics,
                        current_phases=current_phases,
                    )
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        q_values = policy_nets[tls_id](state_tensor).squeeze(0)
                        valid_action_ids = get_valid_action_ids(
                            tls_id,
                            current_phases[tls_id],
                            elapsed_greens[tls_id],
                        )
                        action_id = select_action_from_q_values(
                            q_values, valid_action_ids
                        )
                    action_counts[tls_id][action_id] += 1
                    action_infos[tls_id] = decode_action(tls_id, action_id)

                current_phases, elapsed_greens, switch_plan = apply_actions(
                    action_infos,
                    current_phases,
                    elapsed_greens,
                    metrics=metrics,
                    controlled_lanes=controlled_lanes,
                )
                for tls_id in CONTROLLED_TLS_IDS:
                    if switch_plan[tls_id]["switched"]:
                        switch_counts[tls_id] += 1
                    elif action_infos[tls_id]["type"] == "switch":
                        blocked_switch_counts[tls_id] += 1
        finally:
            traci.close()
    finally:
        if temp_config_path:
            os.unlink(temp_config_path)

    tripinfo_metrics = parse_tripinfo(tripinfo_path)
    os.unlink(tripinfo_path)
    result = finalize_metrics(metrics, tripinfo_metrics)
    result["action_counts"] = {
        tls_id: dict(sorted(counts.items())) for tls_id, counts in action_counts.items()
    }
    result["switch_counts"] = switch_counts
    result["blocked_switch_counts"] = blocked_switch_counts
    return result


def run_max_pressure_controller(route_file=None, end_time=7200):
    """Run the scenario using the MaxPressure greedy baseline (Wei et al., KDD 2019)."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tripinfo_path = tmp.name

    metrics = init_metrics()
    config_path, temp_config_path = create_config_with_route_override(route_file)
    start_traci(build_eval_sumo_cmd(config_path, tripinfo_path, end_time=end_time))
    try:
        try:
            init_phase_lanes()
            controlled_lanes = get_all_controlled_lanes()
            current_phases = {
                tls_id: determine_current_green_phase(tls_id)
                for tls_id in CONTROLLED_TLS_IDS
            }
            elapsed_greens = {tls_id: 0 for tls_id in CONTROLLED_TLS_IDS}

            step_with_metrics(20, metrics, controlled_lanes)

            while (
                traci.simulation.getMinExpectedNumber() > 0
                and traci.simulation.getTime() < end_time
            ):
                action_infos = {}
                for tls_id in CONTROLLED_TLS_IDS:
                    best_phase = select_max_pressure_phase(
                        tls_id, current_phases[tls_id], elapsed_greens[tls_id]
                    )
                    if best_phase == current_phases[tls_id]:
                        action_infos[tls_id] = {"type": "extend", "target_phase": None}
                    else:
                        action_infos[tls_id] = {
                            "type": "switch",
                            "target_phase": best_phase,
                        }

                current_phases, elapsed_greens, _ = apply_actions(
                    action_infos,
                    current_phases,
                    elapsed_greens,
                    metrics=metrics,
                    controlled_lanes=controlled_lanes,
                )
        finally:
            traci.close()
    finally:
        if temp_config_path:
            os.unlink(temp_config_path)

    tripinfo_metrics = parse_tripinfo(tripinfo_path)
    os.unlink(tripinfo_path)
    return finalize_metrics(metrics, tripinfo_metrics)


def evaluate_pair(model_path=MODEL_PATH, route_file=None, end_time=7200):
    """Run fixed-time, MaxPressure, and RL evaluation on the same scenario."""
    return {
        "baseline": run_default_controller(route_file=route_file, end_time=end_time),
        "max_pressure": run_max_pressure_controller(
            route_file=route_file, end_time=end_time
        ),
        "rl": run_rl_controller(
            model_path=model_path, route_file=route_file, end_time=end_time
        ),
    }


def print_comparison(results):
    """Print metrics for fixed-time, MaxPressure, and RL with Δ% vs fixed-time baseline."""
    metric_specs = [
        ("Steps", "steps", False),
        ("Vehicles Arrived", "arrived_vehicles", True),
        ("Avg Queue / Step", "avg_queue_per_step", False),
        ("Avg Lane Wait / Step", "avg_lane_wait_per_step", False),
        ("Avg Trip Duration", "avg_trip_duration", False),
        ("Avg Waiting Time", "avg_waiting_time", False),
        ("Avg Time Loss", "avg_time_loss", False),
        ("Max Waiting Time", "max_waiting_time", False),
    ]

    def pct(base, val, higher_is_better=False):
        if base == 0:
            return "n/a"
        delta = (
            ((val - base) / abs(base) * 100)
            if higher_is_better
            else ((base - val) / abs(base) * 100)
        )
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.1f}%"

    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    has_mp = "max_pressure" in results
    headers = (
        ("Metric", "Fixed-Time", "MaxPressure", "RL", "Δ% (RL vs Fixed)")
        if has_mp
        else ("Metric", "Fixed-Time", "RL", "Δ% (RL vs Fixed)")
    )
    rows = []
    for label, key, higher in metric_specs:
        b = results["baseline"][key]
        r = results["rl"][key]
        if has_mp:
            mp = results["max_pressure"][key]
            rows.append((label, fmt(b), fmt(mp), fmt(r), pct(b, r, higher)))
        else:
            rows.append((label, fmt(b), fmt(r), pct(b, r, higher)))

    n = len(headers)
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(n)]
    sep = "  "
    header_line = sep.join(
        f"{headers[i]:<{widths[i]}}" if i == 0 else f"{headers[i]:>{widths[i]}}"
        for i in range(n)
    )
    divider = sep.join("-" * w for w in widths)
    print(header_line)
    print(divider)
    for row in rows:
        print(
            sep.join(
                f"{row[i]:<{widths[i]}}" if i == 0 else f"{row[i]:>{widths[i]}}"
                for i in range(n)
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the default SUMO controller against the trained RL controller."
    )
    parser.add_argument(
        "--mode",
        choices=("both", "baseline", "max_pressure", "rl"),
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
    parser.add_argument(
        "--end-time",
        type=int,
        default=7200,
        help="Simulated seconds to run each evaluation (default: 7200).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}

    if args.mode in ("both", "baseline"):
        results["baseline"] = run_default_controller(
            route_file=args.route_file, end_time=args.end_time
        )
    if args.mode in ("both", "max_pressure"):
        results["max_pressure"] = run_max_pressure_controller(
            route_file=args.route_file, end_time=args.end_time
        )
    if args.mode in ("both", "rl"):
        results["rl"] = run_rl_controller(
            model_path=args.model_path,
            route_file=args.route_file,
            end_time=args.end_time,
        )

    if args.mode == "both":
        print_comparison(results)
    else:
        controller = args.mode
        for key, value in results[controller].items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()

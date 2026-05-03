"""Inspect traffic lights, phases, and controlled lanes in the active SUMO network."""

import argparse
import sys
from pathlib import Path

import traci

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import CONTROLLED_TLS_IDS, GREEN_PHASES, SUMO_CONFIG, TLS_ID  # noqa: E402


def unique_lanes(tls_id):
    return list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))


def print_tls_ids():
    print("Traffic light IDs:")
    for tls_id in traci.trafficlight.getIDList():
        marker = "controlled" if tls_id in CONTROLLED_TLS_IDS else "uncontrolled"
        print(f" - {tls_id} ({marker})")


def print_phases(tls_id):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    print(f"\nPhases for {tls_id}:")
    for idx, phase in enumerate(logic.phases):
        green_marker = " green" if idx in GREEN_PHASES and tls_id == TLS_ID else ""
        print(
            f" {idx:02d}: duration={phase.duration:>5.1f}s state={phase.state}{green_marker}"
        )


def print_lane_mapping(tls_id):
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]

    print(f"\nControlled lanes for {tls_id}:")
    for idx, lane in enumerate(unique_lanes(tls_id)):
        print(f" {idx:02d}: {lane}")

    print(f"\nPhase-to-lane mapping for {tls_id}:")
    for phase_idx, phase in enumerate(logic.phases):
        active_lanes = set()
        for link_idx, signal_char in enumerate(phase.state):
            if signal_char not in ("G", "g") or link_idx >= len(controlled_links):
                continue
            for connection in controlled_links[link_idx]:
                active_lanes.add(connection[0])

        if active_lanes:
            print(f" {phase_idx:02d}: {sorted(active_lanes)}")


def inspect(args):
    sumo_cmd = ["sumo", "-c", args.config]
    traci.start(sumo_cmd)
    try:
        if args.list_tls:
            print_tls_ids()

        tls_ids = CONTROLLED_TLS_IDS if args.all_controlled else [args.tls_id]
        for tls_id in tls_ids:
            if args.phases:
                print_phases(tls_id)
            if args.lanes:
                print_lane_mapping(tls_id)
    finally:
        traci.close()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the active SUMO traffic-light network."
    )
    parser.add_argument("--config", default=SUMO_CONFIG, help="SUMO config to load.")
    parser.add_argument("--tls-id", default=TLS_ID, help="Traffic light ID to inspect.")
    parser.add_argument(
        "--all-controlled", action="store_true", help="Inspect every controlled TLS."
    )
    parser.add_argument(
        "--list-tls",
        action="store_true",
        help="List every traffic light ID in the network.",
    )
    parser.add_argument(
        "--phases", action="store_true", help="Print phase definitions."
    )
    parser.add_argument(
        "--lanes",
        action="store_true",
        help="Print controlled lanes and phase-lane mappings.",
    )
    args = parser.parse_args()

    if not any([args.list_tls, args.phases, args.lanes]):
        args.list_tls = True
        args.phases = True

    inspect(args)


if __name__ == "__main__":
    main()

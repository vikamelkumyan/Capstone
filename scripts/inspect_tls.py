import sys
from pathlib import Path

import traci

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import SUMO_CONFIG, TLS_ID  # noqa: E402

SUMO_CMD = ["sumo", "-c", SUMO_CONFIG]


def main():
    """Inspect the controlled lanes, links, and phases for the target TLS."""
    traci.start(SUMO_CMD)
    try:
        print("TLS ID:", TLS_ID)
        print()

        lanes = traci.trafficlight.getControlledLanes(TLS_ID)
        links = traci.trafficlight.getControlledLinks(TLS_ID)
        logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]

        unique_lanes = list(dict.fromkeys(lanes))

        print("Controlled lanes:")
        for lane in unique_lanes:
            print(" -", lane)

        print("\nNumber of controlled links:", len(links))
        print("\nPhases:")
        for i, phase in enumerate(logic.phases):
            print(f"{i}: duration={phase.duration}, state={phase.state}")
    finally:
        traci.close()


if __name__ == "__main__":
    main()

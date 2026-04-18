import sys
from pathlib import Path

import traci

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import SUMO_CONFIG, TLS_ID

SUMO_CMD = ["sumo", "-c", SUMO_CONFIG]


def print_phases():
    """Print the exact traffic light phases as defined by SUMO."""
    traci.start(SUMO_CMD)
    try:
        logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]

        print("\n=== TRAFFIC LIGHT PHASES EXACTLY AS DEFINED IN SUMO ===")
        for i, phase in enumerate(logic.phases):
            print(f"Phase {i:02d}: {phase.state} (Dur: {phase.duration}s)")
    finally:
        traci.close()


if __name__ == "__main__":
    print_phases()

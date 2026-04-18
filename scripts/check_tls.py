import sys
from pathlib import Path

import traci

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import SUMO_CONFIG

SUMO_CMD = ["sumo", "-c", SUMO_CONFIG]


def main():
    """Print all traffic light IDs in the configured SUMO scenario."""
    traci.start(SUMO_CMD)
    try:
        tls_ids = traci.trafficlight.getIDList()

        print("\nTraffic Light IDs:")
        for tls in tls_ids:
            print(tls)
    finally:
        traci.close()


if __name__ == "__main__":
    main()

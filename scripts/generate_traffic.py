import argparse
import random
from pathlib import Path

SCENARIOS = {
    "morning_rush": {
        "1292696869#0": 0.40,
        "515004000#2": 0.05,
        "380344432#4": 0.10,
        "1292696982#1": 0.10,
    },
    "evening_rush": {
        "1292696869#0": 0.05,
        "515004000#2": 0.40,
        "380344432#4": 0.15,
        "1292696982#1": 0.15,
    },
    "off_peak": {
        "1292696869#0": 0.02,
        "515004000#2": 0.02,
        "380344432#4": 0.02,
        "1292696982#1": 0.02,
    },
}

VALID_ROUTES = [
    ("1292696869#0", "380344433#2"),
    ("1292696869#0", "1335206306#1"),
    ("515004000#2", "1292696867#1"),
    ("515004000#2", "380344433#2"),
    ("515004000#2", "1335206306#1"),
    ("1292696982#1", "380344435#1"),
    ("1292696982#1", "1292696867#1"),
    ("1292696982#1", "380344433#2"),
    ("1292696982#1", "1335206306#1"),
    ("380344432#4", "1335206306#1"),
    ("380344432#4", "380344435#1"),
]


def generate_route_file(filename, probabilities, seed=42, n_steps=3600):
    """
    Generate a SUMO route file from per-edge departure probabilities.
    """
    rng = random.Random(seed)
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<routes>\n")
        f.write(
            '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="50"/>\n'
        )

        route_map = {}
        for idx, (src, dst) in enumerate(VALID_ROUTES):
            route_id = f"r_{idx}"
            f.write(f'    <route id="{route_id}" edges="{src} {dst}"/>\n')
            if src not in route_map:
                route_map[src] = []
            route_map[src].append(route_id)

        f.write("\n")

        veh_id = 0
        for step in range(n_steps):
            for edge, prob in probabilities.items():
                if edge in route_map and rng.random() < prob:
                    target_route = rng.choice(route_map[edge])
                    f.write(
                        f'    <vehicle id="v_{veh_id}" type="car" route="{target_route}" depart="{step}"/>\n'
                    )
                    veh_id += 1

        f.write("</routes>\n")

    print(f"Generated {veh_id} vehicles in {filename}")


def parse_args():
    """Parse command-line options for traffic generation."""
    parser = argparse.ArgumentParser(
        description="Generate SUMO route demand for the intersection."
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="morning_rush",
        help="Named traffic demand scenario to generate.",
    )
    parser.add_argument(
        "--output",
        default="sumo_data/routes.rou.xml",
        help="Output path for the generated SUMO route file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic route generation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3600,
        help="Number of simulation seconds to populate with traffic demand.",
    )
    return parser.parse_args()


def main():
    """Generate the configured SUMO route file."""
    args = parse_args()
    print(f"Generating scenario '{args.scenario}' with seed {args.seed}...")
    generate_route_file(
        args.output,
        SCENARIOS[args.scenario],
        seed=args.seed,
        n_steps=args.steps,
    )


if __name__ == "__main__":
    main()

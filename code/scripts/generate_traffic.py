import argparse
import random
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

DEFAULT_NET_FILE = "data/raw_data/sumo_data/komitas.net.xml"

PORTALS = {
    "Gyulbenkyan-West": {"edge": "23635021#5", "direction": "west"},
    "Gyulbenkyan-North": {"edge": "515003999#0", "direction": "north"},
    "Vagharshyan-North": {"edge": "749215005#0", "direction": "north"},
    "Vagharshyan-South": {"edge": "380344432#3", "direction": "south"},
    "Papazyan-West": {"edge": "-174418763#1", "direction": "west"},
    "Papazyan-East": {"edge": "1149127548#2", "direction": "east"},
    "Vracakan-North": {"edge": "379124195#8", "direction": "north"},
    "Vracakan-South": {"edge": "481331815#2", "direction": "south"},
    "Griboyedov-West": {"edge": "1291256189#4", "direction": "west"},
    "Tigranyan-North": {"edge": "1291290491#4", "direction": "north"},
    "Tigranyan-East": {"edge": "379128397#1", "direction": "east"},
}

RUSH_HOUR_PROFILE = {
    "Vagharshyan-South": 0.26,
    "Papazyan-East": 0.18,
    "Vracakan-South": 0.22,
    "Tigranyan-East": 0.28,
    "Gyulbenkyan-West": 0.12,
    "Vagharshyan-North": 0.08,
    "Griboyedov-West": 0.12,
}

SCENARIOS = {
    "rush_hour": RUSH_HOUR_PROFILE,
    "off_peak": {
        "Gyulbenkyan-West": 0.04,
        "Gyulbenkyan-North": 0.04,
        "Vagharshyan-North": 0.04,
        "Vagharshyan-South": 0.04,
        "Papazyan-West": 0.04,
        "Papazyan-East": 0.04,
        "Vracakan-North": 0.04,
        "Vracakan-South": 0.04,
        "Griboyedov-West": 0.04,
        "Tigranyan-North": 0.04,
        "Tigranyan-East": 0.04,
    },
    "corridor_stress": {
        "Gyulbenkyan-West": 0.14,
        "Gyulbenkyan-North": 0.28,
        "Vagharshyan-North": 0.24,
        "Vagharshyan-South": 0.24,
        "Papazyan-West": 0.18,
        "Papazyan-East": 0.18,
        "Vracakan-North": 0.18,
        "Vracakan-South": 0.20,
        "Griboyedov-West": 0.24,
        "Tigranyan-North": 0.24,
        "Tigranyan-East": 0.24,
    },
}

PORTAL_EDGES = {portal_name: portal["edge"] for portal_name, portal in PORTALS.items()}


def build_reachable_portal_destinations(net_file):
    """Compute which portal edges are reachable from each portal edge."""
    root = ET.parse(net_file).getroot()
    successors = defaultdict(set)
    for connection in root.findall("connection"):
        source = connection.attrib.get("from")
        target = connection.attrib.get("to")
        if (
            source
            and target
            and not source.startswith(":")
            and not target.startswith(":")
        ):
            successors[source].add(target)

    portal_edges = list(PORTAL_EDGES.values())
    reachable = {}
    for source_edge in portal_edges:
        seen = {source_edge}
        queue = deque([source_edge])
        reachable_destinations = set()

        while queue:
            edge = queue.popleft()
            for successor in successors[edge]:
                if successor in seen:
                    continue
                seen.add(successor)
                queue.append(successor)
                if successor in portal_edges and successor != source_edge:
                    reachable_destinations.add(successor)

        reachable[source_edge] = sorted(reachable_destinations)

    return reachable


def count_routed_vehicles(route_file):
    """Count routed vehicles written by duarouter."""
    root = ET.parse(route_file).getroot()
    return len(root.findall("vehicle"))


def choose_destination_edges(source_portal, candidate_edges):
    """Prefer destinations on a different corridor side to encourage longer trips."""
    source_direction = PORTALS[source_portal]["direction"]
    preferred = [
        edge
        for portal_name, edge in PORTAL_EDGES.items()
        if edge in candidate_edges
        and PORTALS[portal_name]["direction"] != source_direction
    ]
    return preferred or list(candidate_edges)


def build_probe_trip_file(path, source_edge, target_edge):
    """Write a minimal trip file for validating one OD pair with duarouter."""
    with Path(path).open("w", encoding="utf-8") as trip_file:
        trip_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        trip_file.write("<routes>\n")
        trip_file.write(
            '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="50"/>\n'
        )
        trip_file.write(
            f'    <trip id="probe" type="car" from="{source_edge}" to="{target_edge}" depart="0"/>\n'
        )
        trip_file.write("</routes>\n")


def route_pair_is_valid(net_file, source_edge, target_edge):
    """Return True when duarouter can build a route for a source/destination portal pair."""
    with tempfile.NamedTemporaryFile(suffix=".trips.xml", delete=False) as tmp_trip:
        trip_path = Path(tmp_trip.name)
    with tempfile.NamedTemporaryFile(suffix=".rou.xml", delete=False) as tmp_route:
        route_path = Path(tmp_route.name)

    try:
        build_probe_trip_file(trip_path, source_edge, target_edge)
        result = subprocess.run(
            [
                "duarouter",
                "-n",
                str(net_file),
                "-r",
                str(trip_path),
                "-o",
                str(route_path),
                "--no-warnings",
                "--no-step-log",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return (
            result.returncode == 0
            and route_path.exists()
            and count_routed_vehicles(route_path) > 0
        )
    finally:
        if trip_path.exists():
            trip_path.unlink()
        if route_path.exists():
            route_path.unlink()


@lru_cache(maxsize=8)
def build_valid_portal_destinations(net_file):
    """Return duarouter-validated destination portals for each source portal edge."""
    net_path = Path(net_file)
    reachable = build_reachable_portal_destinations(net_path)
    valid_destinations = {}

    for source_portal, source_edge in PORTAL_EDGES.items():
        candidate_edges = choose_destination_edges(
            source_portal,
            reachable.get(source_edge, []),
        )
        valid_edges = [
            target_edge
            for target_edge in candidate_edges
            if route_pair_is_valid(net_path, source_edge, target_edge)
        ]
        if not valid_edges:
            valid_edges = [
                target_edge
                for target_edge in reachable.get(source_edge, [])
                if route_pair_is_valid(net_path, source_edge, target_edge)
            ]
        valid_destinations[source_edge] = valid_edges

    return valid_destinations


def generate_route_file(
    filename, probabilities, seed=42, n_steps=3600, net_file=DEFAULT_NET_FILE
):
    """Generate a SUMO route file by building trips and routing them with duarouter."""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net_path = Path(net_file)
    if not net_path.is_absolute():
        net_path = net_path.resolve()
    valid_destinations = build_valid_portal_destinations(str(net_path))

    with tempfile.NamedTemporaryFile(suffix=".trips.xml", delete=False) as tmp:
        trip_path = Path(tmp.name)

    try:
        trip_count = write_trip_file_reachable(
            trip_path,
            probabilities,
            valid_destinations,
            seed=seed,
            n_steps=n_steps,
        )
        subprocess.run(
            [
                "duarouter",
                "-n",
                str(net_path),
                "-r",
                str(trip_path),
                "-o",
                str(output_path),
                "--ignore-errors",
            ],
            check=True,
        )
        routed_vehicle_count = count_routed_vehicles(output_path)
        alt_output = Path(str(output_path).replace(".rou.xml", ".rou.alt.xml"))
        if alt_output.exists():
            alt_output.unlink()
    finally:
        if trip_path.exists():
            trip_path.unlink()

    print(
        f"Generated {routed_vehicle_count} routed vehicles in {filename}"
        f" from {trip_count} attempted trips"
    )


def scale_probabilities(probabilities, factor):
    """Scale scenario probabilities while keeping them in [0, 1]."""
    return {
        portal_name: min(1.0, max(0.0, prob * factor))
        for portal_name, prob in probabilities.items()
    }


def write_trip_file_reachable(
    path, probabilities, reachable_destinations, seed=42, n_steps=3600
):
    """Write only trips whose destination portal is reachable from the source portal."""
    rng = random.Random(seed)
    trip_count = 0

    with Path(path).open("w", encoding="utf-8") as trip_file:
        trip_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        trip_file.write("<routes>\n")
        trip_file.write(
            '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="50"/>\n'
        )

        for step in range(n_steps):
            for portal_name, prob in probabilities.items():
                source_edge = PORTAL_EDGES[portal_name]
                candidate_destinations = reachable_destinations.get(source_edge, [])
                if not candidate_destinations:
                    continue
                if rng.random() < prob:
                    target_edge = rng.choice(candidate_destinations)
                    trip_file.write(
                        f'    <trip id="t_{trip_count}" type="car" from="{source_edge}" to="{target_edge}" depart="{step}"/>\n'
                    )
                    trip_count += 1

        trip_file.write("</routes>\n")

    return trip_count


def parse_args():
    """Parse command-line options for traffic generation."""
    parser = argparse.ArgumentParser(
        description="Generate SUMO route demand for the Komitas corridor."
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="off_peak",
        help="Named traffic demand scenario to generate.",
    )
    parser.add_argument(
        "--output",
        default="data/raw_data/sumo_data/routes.rou.xml",
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
    parser.add_argument(
        "--net-file",
        default=DEFAULT_NET_FILE,
        help="SUMO network used by duarouter to compute valid full routes.",
    )
    parser.add_argument(
        "--demand-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the scenario spawn probabilities.",
    )
    return parser.parse_args()


def main():
    """Generate the configured SUMO route file."""
    args = parse_args()
    scaled_probabilities = scale_probabilities(
        SCENARIOS[args.scenario],
        args.demand_scale,
    )
    print(
        f"Generating scenario '{args.scenario}' with seed {args.seed}"
        f" and demand scale {args.demand_scale}..."
    )
    generate_route_file(
        args.output,
        scaled_probabilities,
        seed=args.seed,
        n_steps=args.steps,
        net_file=args.net_file,
    )


if __name__ == "__main__":
    main()

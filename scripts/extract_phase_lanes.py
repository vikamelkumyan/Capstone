import traci

SUMO_CMD = ["sumo-gui", "-c", "sumo_data/komitas-vagharshyan.sumocfg"]
TLS_ID = "cluster_11668441165_11668441166_11668441167_2912634528_#8more"

GREEN_PHASES = [0, 2, 4, 6]


def get_unique_controlled_lanes():
    lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    return list(dict.fromkeys(lanes))


def main():
    traci.start(SUMO_CMD)

    print("\n=== CONTROLLED LANES ===")
    lanes = get_unique_controlled_lanes()
    for i, lane in enumerate(lanes):
        print(f"{i}: {lane}")

    print("\n=== CONTROLLED LINKS ===")
    controlled_links = traci.trafficlight.getControlledLinks(TLS_ID)

    for idx, links in enumerate(controlled_links):
        print(f"\nLink index {idx}:")
        if not links:
            print("  None")
            continue

        for connection in links:
            # connection is usually (inLane, outLane, viaLane)
            print(f"  {connection}")

    print("\n=== PHASE -> ACTIVE LINKS / LANES ===")
    logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]

    for phase_idx, phase in enumerate(logic.phases):
        state = phase.state
        print(f"\nPhase {phase_idx}: state = {state}")

        active_link_indices = []
        active_in_lanes = set()

        for link_idx, signal_char in enumerate(state):
            if signal_char in ("G", "g"):
                active_link_indices.append(link_idx)

                if link_idx < len(controlled_links):
                    links = controlled_links[link_idx]
                    if links:
                        for connection in links:
                            in_lane = connection[0]
                            active_in_lanes.add(in_lane)

        print(f"  Active green links: {active_link_indices}")
        print(f"  Active input lanes: {sorted(active_in_lanes)}")

    print("\n=== GREEN PHASE SUMMARY ===")
    for phase_idx in GREEN_PHASES:
        phase = logic.phases[phase_idx]
        state = phase.state

        active_in_lanes = set()

        for link_idx, signal_char in enumerate(state):
            if signal_char in ("G", "g"):
                if link_idx < len(controlled_links):
                    links = controlled_links[link_idx]
                    if links:
                        for connection in links:
                            in_lane = connection[0]
                            active_in_lanes.add(in_lane)

        print(f"Phase {phase_idx}: {sorted(active_in_lanes)}")

    traci.close()


if __name__ == "__main__":
    main()
import traci

SUMO_CMD = ["sumo", "-c", "sumo_data/komitas-vagharshyan.sumocfg"]
TLS_ID = "cluster_11668441165_11668441166_11668441167_2912634528_#8more"

traci.start(SUMO_CMD)

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

traci.close()
import traci

SUMO_CMD = ["sumo", "-c", "sumo_data/komitas-vagharshyan.sumocfg"]

traci.start(SUMO_CMD)

tls_ids = traci.trafficlight.getIDList()

print("\nTraffic Light IDs:")
for tls in tls_ids:
    print(tls)

traci.close()
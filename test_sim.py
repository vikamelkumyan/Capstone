import traci

SUMO_CMD = ["sumo", "-c", "sumo_data/komitas-vagharshyan.sumocfg"]

traci.start(SUMO_CMD)

for step in range(100):
    traci.simulationStep()
    if step % 10 == 0:
        print(
            f"step={step}, "
            f"loaded={traci.simulation.getLoadedNumber()}, "
            f"departed={traci.simulation.getDepartedNumber()}, "
            f"arrived={traci.simulation.getArrivedNumber()}, "
            f"min_expected={traci.simulation.getMinExpectedNumber()}"
        )

traci.close()
import random
import numpy as np
import traci

SUMO_CMD = ["sumo-gui", "-c", "sumo_data/komitas-vagharshyan.sumocfg", "--start"]
TLS_ID = "cluster_11668441165_11668441166_11668441167_2912634528_#8more"

GREEN_PHASES = [0, 2, 4, 6]

# -------------------------------
# Phase -> lane mapping (YOUR DATA)
# -------------------------------
PHASE_LANES = {
    0: [
        '380344432#4_0', '380344432#4_1', '380344432#4_2', '380344432#4_3',
        '515004000#2_0', '515004000#2_1', '515004000#2_2', '515004000#2_3'
    ],
    2: [
        '515004000#2_0', '515004000#2_1', '515004000#2_2', '515004000#2_3'
    ],
    4: [
        '1292696869#0_0', '1292696869#0_1',
        '1292696982#1_0', '1292696982#1_1', '1292696982#1_2', '1292696982#1_3'
    ],
    6: [
        '1292696982#1_0', '1292696982#1_1',
        '1292696982#1_2', '1292696982#1_3',
        '380344432#4_0'
    ],
}

# -------------------------------
# PARAMETERS
# -------------------------------
EXTEND_STEP = 5
YELLOW_DURATION = 3
MAX_GREEN = 40

# -------------------------------
# HELPERS
# -------------------------------
def step_for_seconds(seconds):
    for _ in range(seconds):
        traci.simulationStep()


def get_next_phase(current):
    idx = GREEN_PHASES.index(current)
    return GREEN_PHASES[(idx + 1) % len(GREEN_PHASES)]


def get_other_phases(current):
    next_p = get_next_phase(current)
    return [p for p in GREEN_PHASES if p not in (current, next_p)]


# -------------------------------
# DEMAND (queue only for stability)
# -------------------------------
def get_phase_demand(phase):
    lanes = set(PHASE_LANES[phase])
    return sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)


# -------------------------------
# DISCRETIZATION
# -------------------------------
def discretize_d(d):
    if d <= 3:
        return 0
    elif d <= 8:
        return 1
    elif d <= 15:
        return 2
    else:
        return 3


def discretize_t(t):
    return min(t // 5, 7)


# -------------------------------
# STATE
# -------------------------------
def get_state(current_phase, elapsed_green):
    next_phase = get_next_phase(current_phase)
    other_phases = get_other_phases(current_phase)

    d_current = get_phase_demand(current_phase)
    d_next = get_phase_demand(next_phase)
    d_others = max(get_phase_demand(p) for p in other_phases)

    return (
        discretize_d(d_current),
        discretize_d(d_next),
        discretize_d(d_others),
        discretize_t(elapsed_green)
    )


def state_to_index(state):
    a, b, c, d = state
    return a * 4 * 4 * 8 + b * 4 * 8 + c * 8 + d


# -------------------------------
# REWARD
# -------------------------------
def get_reward():
    lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    lanes = list(set(lanes))

    total_wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
    total_halted = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

    return -(0.5 * total_wait + 4.0 * total_halted)


# -------------------------------
# SWITCH LOGIC
# -------------------------------
def switch_phase(current_phase):
    next_phase = get_next_phase(current_phase)

    # yellow transition
    traci.trafficlight.setPhase(TLS_ID, current_phase + 1)
    step_for_seconds(YELLOW_DURATION)

    # new green
    traci.trafficlight.setPhase(TLS_ID, next_phase)

    return next_phase


# -------------------------------
# TRAIN
# -------------------------------
def train():
    n_states = 4 * 4 * 4 * 8
    n_actions = 2  # extend / switch

    q_table = np.zeros((n_states, n_actions))

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1

    episodes = 10
    decisions_per_episode = 100

    for episode in range(episodes):
        traci.start(SUMO_CMD)

        current_phase = traci.trafficlight.getPhase(TLS_ID)
        elapsed_green = 0

        step_for_seconds(20)

        episode_reward = 0

        for step in range(decisions_per_episode):
            state = get_state(current_phase, elapsed_green)
            s_idx = state_to_index(state)

            # epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                action = int(np.argmax(q_table[s_idx]))

            # ---------------- ACTION ----------------
            if action == 0:  # extend
                step_for_seconds(EXTEND_STEP)
                elapsed_green += EXTEND_STEP

                # force switch if too long
                if elapsed_green >= MAX_GREEN:
                    current_phase = switch_phase(current_phase)
                    elapsed_green = 0

            else:  # switch
                current_phase = switch_phase(current_phase)
                elapsed_green = 0
                step_for_seconds(EXTEND_STEP)

            # ---------------- UPDATE ----------------
            reward = get_reward()
            next_state = get_state(current_phase, elapsed_green)
            ns_idx = state_to_index(next_state)

            q_table[s_idx, action] += alpha * (
                reward + gamma * np.max(q_table[ns_idx]) - q_table[s_idx, action]
            )

            episode_reward += reward

        traci.close()
        print(f"Episode {episode+1}: total reward = {episode_reward:.2f}")

    np.save("q_table.npy", q_table)
    print("Training complete!")


if __name__ == "__main__":
    train()
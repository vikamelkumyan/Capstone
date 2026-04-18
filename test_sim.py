import torch
import traci
from train import (
    DQN, init_phase_lanes, get_state, switch_phase, step_for_seconds,
    EXTEND_STEP, MAX_GREEN, MIN_GREEN, MODEL_PATH, TLS_ID, SUMO_CMD
)


def test():
    """Run the trained controller in SUMO GUI mode."""
    state_dim = 4
    action_dim = 2

    policy_net = DQN(state_dim, action_dim)
    try:
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        policy_net.eval()
        print(f"Successfully loaded trained DQN model ({MODEL_PATH}).")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Please run train.py first.")
        return

    traci.start(SUMO_CMD)
    try:
        init_phase_lanes()

        current_phase = traci.trafficlight.getPhase(TLS_ID)
        elapsed_green = 0

        step_for_seconds(20)

        decisions = 200
        render_delay = 0.5

        for step in range(decisions):
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("All vehicles have departed/arrived. Ending simulation early.")
                break

            state = get_state(current_phase, elapsed_green)

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

            action_name = "EXTEND" if action == 0 else "SWITCH"
            print(
                f"Step {step+1:03d} | State: {state.tolist()} | "
                f"Q-values: {q_values.tolist()[0]} | Action: {action_name}"
            )

            if action == 0:
                step_for_seconds(EXTEND_STEP, render_delay=render_delay)
                elapsed_green += EXTEND_STEP

                if elapsed_green >= MAX_GREEN:
                    print(" -> MAX_GREEN duration limit reached. Intervening to force switch.")
                    current_phase = switch_phase(current_phase, render_delay=render_delay)
                    elapsed_green = 0
            else:
                if elapsed_green < MIN_GREEN:
                    print(" -> MIN_GREEN not yet reached. Overriding SWITCH to EXTEND.")
                    step_for_seconds(EXTEND_STEP, render_delay=render_delay)
                    elapsed_green += EXTEND_STEP
                else:
                    current_phase = switch_phase(current_phase, render_delay=render_delay)
                    elapsed_green = 0
                    step_for_seconds(EXTEND_STEP, render_delay=render_delay)
    finally:
        traci.close()

    print("Testing visualization complete!")


if __name__ == "__main__":
    test()

import argparse
import os

import torch
import traci

from train import (
    ACTION_DIM,
    DQN,
    EXTEND_STEP,
    MAX_GREEN,
    MIN_GREEN,
    MODEL_PATH,
    STATE_DIM,
    TLS_ID,
    build_sumo_cmd,
    create_config_with_route_override,
    decode_action,
    get_state,
    init_phase_lanes,
    switch_phase,
    switch_to_phase,
    step_for_seconds,
)


def load_policy(model_path):
    """Load the trained controller for GUI testing."""
    policy_net = DQN(STATE_DIM, ACTION_DIM)
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location="cpu"))
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please run train.py first.")
        return None
    except RuntimeError as exc:
        print(
            "Error: saved model is incompatible with the current controller "
            "architecture. Retrain with train.py and try again."
        )
        print(f"Details: {exc}")
        return None

    policy_net.eval()
    print(f"Successfully loaded trained DQN model ({model_path}).")
    return policy_net


def test(model_path=MODEL_PATH, route_file=None, decisions=200, render_delay=0.5):
    """Run the trained controller in SUMO GUI mode."""
    policy_net = load_policy(model_path)
    if policy_net is None:
        return

    config_path, temp_config_path = create_config_with_route_override(route_file)
    sumo_cmd = build_sumo_cmd(config_path, sumo_binary="sumo-gui")

    traci.start(sumo_cmd)
    try:
        init_phase_lanes()

        current_phase = traci.trafficlight.getPhase(TLS_ID)
        elapsed_green = 0

        step_for_seconds(20, render_delay=render_delay)

        for step in range(decisions):
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("All vehicles have departed/arrived. Ending simulation early.")
                break

            state = get_state(current_phase, elapsed_green)

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

            action_info = decode_action(action)
            action_name = (
                "EXTEND"
                if action_info["type"] == "extend"
                else f"SWITCH->{action_info['target_phase']}"
            )
            print(
                f"Step {step + 1:03d} | State: {state.tolist()} | "
                f"Q-values: {q_values.tolist()[0]} | Action: {action_name}"
            )

            if action_info["type"] == "extend":
                step_for_seconds(EXTEND_STEP, render_delay=render_delay)
                elapsed_green += EXTEND_STEP

                if elapsed_green >= MAX_GREEN:
                    print(
                        " -> MAX_GREEN duration limit reached. Intervening to force switch."
                    )
                    current_phase = switch_phase(
                        current_phase, render_delay=render_delay
                    )
                    elapsed_green = 0
            else:
                target_phase = action_info["target_phase"]
                if target_phase == current_phase:
                    print(
                        " -> Target phase is already active. Treating action as EXTEND."
                    )
                    step_for_seconds(EXTEND_STEP, render_delay=render_delay)
                    elapsed_green += EXTEND_STEP
                elif elapsed_green < MIN_GREEN:
                    print(" -> MIN_GREEN not yet reached. Overriding SWITCH to EXTEND.")
                    step_for_seconds(EXTEND_STEP, render_delay=render_delay)
                    elapsed_green += EXTEND_STEP
                else:
                    current_phase = switch_to_phase(
                        current_phase,
                        target_phase,
                        render_delay=render_delay,
                    )
                    elapsed_green = 0
                    step_for_seconds(EXTEND_STEP, render_delay=render_delay)
    finally:
        traci.close()
        if temp_config_path:
            os.unlink(temp_config_path)

    print("Testing visualization complete!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the trained controller in SUMO GUI."
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to the trained model used for visualization.",
    )
    parser.add_argument(
        "--route-file",
        default=None,
        help="Optional route file override used for the GUI run.",
    )
    parser.add_argument(
        "--decisions",
        type=int,
        default=200,
        help="Maximum number of controller decisions to visualize.",
    )
    parser.add_argument(
        "--render-delay",
        type=float,
        default=0.5,
        help="Delay in seconds between SUMO GUI steps.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(
        model_path=args.model_path,
        route_file=args.route_file,
        decisions=args.decisions,
        render_delay=args.render_delay,
    )

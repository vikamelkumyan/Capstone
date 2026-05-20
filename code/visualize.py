import argparse
import os

import torch
import traci

from train import (
    CONTROLLED_TLS_IDS,
    MODEL_PATH,
    apply_actions,
    build_sumo_cmd,
    create_config_with_route_override,
    decode_action,
    determine_current_green_phase,
    get_congestion_metrics,
    get_state,
    get_valid_action_ids,
    init_phase_lanes,
    load_model_bundle,
    select_action_from_q_values,
    start_traci,
)


def test(
    model_path=MODEL_PATH,
    route_file=None,
    decisions=200,
    render_delay=0.5,
    verbose=False,
):
    """Run the trained multi-intersection controller in SUMO GUI mode."""
    try:
        policy_nets = load_model_bundle(model_path)
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please run train.py first.")
        return
    except RuntimeError as exc:
        print(str(exc))
        return

    print(f"Successfully loaded trained DQN model bundle ({model_path}).")

    config_path, temp_config_path = create_config_with_route_override(route_file)
    sumo_cmd = build_sumo_cmd(config_path, sumo_binary="sumo-gui")

    start_traci(sumo_cmd)
    try:
        init_phase_lanes()
        current_phases = {
            tls_id: determine_current_green_phase(tls_id)
            for tls_id in CONTROLLED_TLS_IDS
        }
        elapsed_greens = {tls_id: 0 for tls_id in CONTROLLED_TLS_IDS}

        for _ in range(20):
            traci.simulationStep()
            if render_delay > 0.0:
                import time

                time.sleep(render_delay)

        for step in range(decisions):
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("All vehicles have departed/arrived. Ending simulation early.")
                break

            local_metrics = {
                tls_id: get_congestion_metrics(tls_id) for tls_id in CONTROLLED_TLS_IDS
            }
            action_infos = {}
            print(f"\nDecision {step + 1:03d}")
            for tls_id in CONTROLLED_TLS_IDS:
                state = get_state(
                    tls_id,
                    current_phases[tls_id],
                    elapsed_greens[tls_id],
                    metrics_by_tls=local_metrics,
                    current_phases=current_phases,
                )
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_nets[tls_id](state_tensor)
                    valid_action_ids = get_valid_action_ids(
                        tls_id,
                        current_phases[tls_id],
                        elapsed_greens[tls_id],
                    )
                    action_id = select_action_from_q_values(
                        q_values.squeeze(0),
                        valid_action_ids,
                    )

                action_info = decode_action(tls_id, action_id)
                action_infos[tls_id] = action_info
                action_name = (
                    "EXTEND"
                    if action_info["type"] == "extend"
                    else f"SWITCH->{action_info['target_phase']}"
                )
                if verbose:
                    print(
                        f"  {tls_id}: phase={current_phases[tls_id]} elapsed={elapsed_greens[tls_id]} "
                        f"state={state.tolist()} q={q_values.tolist()[0]} action={action_name}"
                    )
                else:
                    metrics = local_metrics[tls_id]
                    print(
                        f"  {tls_id}: phase={current_phases[tls_id]} "
                        f"elapsed={elapsed_greens[tls_id]} queue={metrics['total_queue']} "
                        f"wait={metrics['total_wait']:.1f} action={action_name}"
                    )

            current_phases, elapsed_greens, _ = apply_actions(
                action_infos,
                current_phases,
                elapsed_greens,
                render_delay=render_delay,
            )
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full state vectors and Q-values for every decision.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(
        model_path=args.model_path,
        route_file=args.route_file,
        decisions=args.decisions,
        render_delay=args.render_delay,
        verbose=args.verbose,
    )

"""Stage 4: online responsibility-aware trajectory prediction demo (paper
Sec. V-C): for one scenario and queried agent, estimate its responsibility
level from windows that have already closed, sample candidate rollouts at
the prediction step, and select the one maximizing that level's learned
reward -- reported against the "highest model log-probability" baseline.

The protocol is deliberately causal. beta at context step k is computed
from the logged future over [k+1, k+metric_horizon], so it is only known
at k+metric_horizon. Predicting at step P therefore uses only windows with
k <= P - metric_horizon; otherwise the level would be derived from the very
future the prediction is scored against.

Example:
    python -m scripts.responsibility.predict_demo \
        --ckpt /path/to/smart.ckpt \
        --data-root /path/to/cache/validation \
        --hmm logs/responsibility/hmm.pkl \
        --reward-nets logs/responsibility/reward_nets.pt \
        --scenario-index 0 --predict-step 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.responsibility.dataset_builder import (  # noqa: E402
    causal_window_steps,
    compute_scenario_responsibility,
)
from scripts.responsibility._cli import (  # noqa: E402
    add_model_args,
    add_responsibility_args,
    responsibility_config_from_args,
)
from src.responsibility.metrics import ground_truth_boxes  # noqa: E402
from src.responsibility.motion_set import MotionSetSynthesizer, shifted_context  # noqa: E402
from src.responsibility.pipeline import (  # noqa: E402
    compose_responsibility_config,
    default_sampling_scheme,
    load_hmm,
    load_reward_nets,
    load_scenario,
    load_smart_model,
    resolve_device,
    scenario_id_of,
)
from src.responsibility.selection import predict_responsibility_aware  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--hmm", required=True)
    p.add_argument("--reward-nets", required=True)
    p.add_argument("--scenario-index", type=int, default=0)
    p.add_argument("--agent-index", type=int, default=None,
                   help="Defaults to the self-driving car, else the first observed vehicle.")
    p.add_argument(
        "--predict-step",
        type=int,
        default=50,
        help="Context step to predict from. Must leave room for at least one "
        "closed responsibility window before it and some logged future after.",
    )
    p.add_argument("--n-candidates", type=int, default=25)
    add_model_args(p)
    add_responsibility_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    device = resolve_device(args.device)
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=device)
    hmm = load_hmm(args.hmm)
    reward_nets = load_reward_nets(args.reward_nets)
    data = load_scenario(args.data_root, args.scenario_index, device)

    num_historical_steps = hydra_cfg.model.model_config.decoder.num_historical_steps
    rcfg = responsibility_config_from_args(args, num_historical_steps)
    synthesizer = MotionSetSynthesizer(
        model,
        default_sampling_scheme(num_k=args.motion_set_num_k, temp=args.motion_set_temp),
        rollout_batch_size=args.rollout_batch_size,
    )

    if args.agent_index is not None:
        query_idx = args.agent_index
    else:
        observed = data["agent"]["valid_mask"][:, args.predict_step]
        sdc = data["agent"]["role"][:, 0] & observed
        vehicles = (data["agent"]["type"] == 0) & observed
        pick = sdc if sdc.any() else vehicles
        query_idx = int(pick.nonzero(as_tuple=True)[0][0])
    role = "SDC" if bool(data["agent"]["role"][query_idx, 0]) else "agent"
    print(f"Scenario {scenario_id_of(data)}, {role} index {query_idx}")

    # --- responsibility history: only windows already closed at the prediction step
    steps = causal_window_steps(data, rcfg, synthesizer, args.predict_step, rcfg.window_stride)
    if len(steps) == 0:
        raise ValueError(
            f"No responsibility window closes at or before step {args.predict_step}. "
            f"Raise --predict-step (needs >= {num_historical_steps - 1 + args.metric_horizon}) "
            f"or shorten --metric-horizon."
        )
    observations = compute_scenario_responsibility(
        synthesizer, data, rcfg, agent_type_filter=None, steps=steps, agents=[query_idx]
    )
    history = sorted(
        (o for o in observations if o.agent_idx == query_idx and o.scope == "neighbours"),
        key=lambda o: o.k,
    )
    if not history:
        raise ValueError(f"agent {query_idx} is not observed at any closed window {steps}")
    beta_history = torch.stack([o.beta for o in history])
    print("beta history (closed windows only):")
    for o in history:
        critical = max(o.per_neighbour.items(), key=lambda kv: kv[1][0], default=(None, None))[0]
        print(f"  k={o.k:3d}  safety={o.beta[0]:.3f}  courtesy={o.beta[1]:.3f}  "
              f"neighbours={len(o.per_neighbour)}  most safety-relevant={critical}")

    # --- prediction at the (later) prediction step
    with shifted_context(data, args.predict_step, num_historical_steps) as shifted:
        prediction = predict_responsibility_aware(
            synthesizer,
            shifted,
            query_idx,
            beta_history,
            hmm,
            reward_nets,
            rcfg,
            n_candidates=args.n_candidates,
        )
        horizon = prediction["horizon"]
        gt = ground_truth_boxes(shifted, [query_idx], num_historical_steps, horizon)
        gt_traj, gt_valid = gt.pos[0].clone(), gt.valid[0].clone()

    print(f"Responsibility level at step {args.predict_step}: {prediction['level']}")

    def _displacement_errors(traj):
        err = torch.linalg.norm(traj[:horizon] - gt_traj, dim=-1)
        if not gt_valid.any():
            return float("nan"), float("nan")
        last_valid = int((torch.arange(horizon, device=gt_valid.device) * gt_valid).argmax())
        return err[gt_valid].mean().item(), err[last_valid].item()

    selected_ade, selected_fde = _displacement_errors(prediction["selected_traj"])
    # baseline: the candidate the model itself considers most likely
    # (features[:, -1] is log_prob, see reward_model.extract_trajectory_features)
    baseline_idx = int(prediction["features"][:, -1].argmax())
    baseline_ade, baseline_fde = _displacement_errors(prediction["candidates"][baseline_idx])

    print(f"Responsibility-aware selection  ADE {selected_ade:.3f} m  FDE {selected_fde:.3f} m")
    print(f"Highest-log-prob baseline       ADE {baseline_ade:.3f} m  FDE {baseline_fde:.3f} m")


if __name__ == "__main__":
    main()

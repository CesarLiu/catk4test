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
from src.responsibility.metrics import ResponsibilityConfig, ground_truth_boxes  # noqa: E402
from src.responsibility.motion_set import MotionSetSynthesizer, shifted_context  # noqa: E402
from src.responsibility.pipeline import (  # noqa: E402
    build_dataloader,
    compose_responsibility_config,
    default_sampling_scheme,
    load_hmm,
    load_reward_nets,
    load_smart_model,
)
from src.responsibility.selection import predict_responsibility_aware  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--hmm", required=True)
    p.add_argument("--reward-nets", required=True)
    p.add_argument("--model-config", default="smart")
    p.add_argument("--scenario-index", type=int, default=0)
    p.add_argument("--agent-index", type=int, default=None, help="Defaults to the first valid vehicle.")
    p.add_argument(
        "--predict-step",
        type=int,
        default=50,
        help="Context step to predict from. Must leave room for at least one "
        "closed responsibility window before it and some logged future after.",
    )
    p.add_argument("--window-stride", type=int, default=5)
    p.add_argument("--metric-horizon", type=int, default=20)
    p.add_argument("--n-candidates", type=int, default=25)
    p.add_argument("--motion-set-num-k", type=int, default=2048)
    p.add_argument("--motion-set-temp", type=float, default=1.0)
    p.add_argument("--rollout-batch-size", type=int, default=8)
    p.add_argument("--neighbor-future", default="logged",
                   choices=["logged", "hidden", "simulated"],
                   help="How much of the neighbours' future the queried agent may "
                        "condition on: logged (they replay their log and stay visible), "
                        "hidden (visible only up to the current step; the paper's "
                        "open-loop reading), simulated (nobody is teacher-forced).")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=args.device)
    hmm = load_hmm(args.hmm)
    reward_nets = load_reward_nets(args.reward_nets)

    dataloader = build_dataloader(args.data_root)
    data = None
    for i, d in enumerate(dataloader):
        if i == args.scenario_index:
            data = d
            break
    if data is None:
        raise ValueError(f"data-root has fewer than {args.scenario_index + 1} scenarios.")

    num_historical_steps = hydra_cfg.model.model_config.decoder.num_historical_steps
    rcfg = ResponsibilityConfig(
        num_historical_steps=num_historical_steps,
        metric_horizon=args.metric_horizon,
        window_stride=args.window_stride,
        neighbor_future=args.neighbor_future,
    )
    synthesizer = MotionSetSynthesizer(
        model,
        default_sampling_scheme(num_k=args.motion_set_num_k, temp=args.motion_set_temp),
        rollout_batch_size=args.rollout_batch_size,
    )

    if args.agent_index is not None:
        query_idx = args.agent_index
    else:
        veh_mask = (data["agent"]["type"] == 0) & data["agent"]["valid_mask"][:, args.predict_step]
        query_idx = int(veh_mask.nonzero(as_tuple=True)[0][0])
    print(f"Scenario {data.get('scenario_id', args.scenario_index)}, agent index {query_idx}")

    # --- responsibility history: only windows already closed at the prediction step
    steps = causal_window_steps(data, rcfg, synthesizer, args.predict_step, args.window_stride)
    if len(steps) == 0:
        raise ValueError(
            f"No responsibility window closes at or before step {args.predict_step}. "
            f"Raise --predict-step (needs >= {num_historical_steps - 1 + args.metric_horizon}) "
            f"or shorten --metric-horizon."
        )
    observations = compute_scenario_responsibility(
        synthesizer, data, rcfg, agent_type_filter=None, steps=steps
    )
    beta_history = torch.stack([o.beta for o in observations if o.agent_idx == query_idx])
    print(f"beta history over context steps {steps}:")
    for k, beta in zip(steps, beta_history):
        print(f"  k={k:3d}  safety={beta[0]:.3f}  courtesy={beta[1]:.3f}")

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
        last_valid = int((torch.arange(horizon) * gt_valid).argmax())
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

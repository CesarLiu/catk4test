"""Stage 4: online responsibility-aware trajectory prediction demo (paper
Sec. V-C): for one scenario and queried agent, compute its current
responsibility level, sample candidate rollouts, and select the one that
maximizes that level's learned reward -- reporting displacement error
against the naive "highest model log-probability" baseline (an ablation
matching the paper's BITS-heuristic comparison in spirit).

Example:
    python -m scripts.responsibility.predict_demo \
        --ckpt /path/to/smart.ckpt \
        --data-root /path/to/cache/validation \
        --hmm logs/responsibility/hmm.pkl \
        --reward-nets logs/responsibility/reward_nets.pt \
        --scenario-index 0 --agent-index 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.responsibility.metrics import (  # noqa: E402
    ResponsibilityConfig,
    ground_truth_boxes,
)
from src.responsibility.motion_set import MotionSetSynthesizer  # noqa: E402
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
    p.add_argument("--n-candidates", type=int, default=25)
    p.add_argument("--motion-set-num-k", type=int, default=2048)
    p.add_argument("--motion-set-temp", type=float, default=1.0)
    p.add_argument("--rollout-batch-size", type=int, default=8)
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
    at_step = num_historical_steps - 1
    if args.agent_index is not None:
        query_idx = args.agent_index
    else:
        veh_mask = (data["agent"]["type"] == 0) & data["agent"]["valid_mask"][:, at_step]
        query_idx = int(veh_mask.nonzero(as_tuple=True)[0][0])
    print(f"Scenario {data.get('scenario_id', args.scenario_index)}, queried agent index {query_idx}")

    synthesizer = MotionSetSynthesizer(
        model,
        default_sampling_scheme(num_k=args.motion_set_num_k, temp=args.motion_set_temp),
        rollout_batch_size=args.rollout_batch_size,
    )
    rcfg = ResponsibilityConfig(num_historical_steps=num_historical_steps)

    from src.responsibility.metrics import compute_agent_responsibility
    from src.responsibility.motion_set import find_neighbors

    tokenized_map, tokenized_agent = synthesizer.tokenize(data)
    n_agent = data["agent"]["position"].shape[0]
    neighbor_indices = find_neighbors(
        data["agent"]["position"][..., :2], data["agent"]["valid_mask"], query_idx, at_step=at_step
    ).nonzero(as_tuple=True)[0].tolist()
    result = compute_agent_responsibility(
        synthesizer, data, tokenized_map, tokenized_agent, query_idx, neighbor_indices, list(range(n_agent)), rcfg
    )
    beta_history = result["beta"].unsqueeze(0)  # [1, 2]: single-step context (see dataset_builder docstring)
    print(f"beta = {beta_history.squeeze(0).tolist()} (safety, courtesy)")

    prediction = predict_responsibility_aware(
        synthesizer,
        data,
        query_idx,
        beta_history,
        hmm,
        reward_nets,
        rcfg,
        n_candidates=args.n_candidates,
    )
    print(f"Responsibility level: {prediction['level']}")

    horizon = prediction["horizon"]
    gt = ground_truth_boxes(data, [query_idx], rcfg.num_historical_steps, horizon)
    gt_traj, gt_valid = gt.pos[0], gt.valid[0]

    def _ade(traj):
        err = torch.linalg.norm(traj[:horizon] - gt_traj, dim=-1)
        return err[gt_valid].mean().item() if gt_valid.any() else float("nan")

    selected_ade = _ade(prediction["selected_traj"])
    # baseline: pick the candidate with the highest model log-probability
    # (features[:, -1] is log_prob, see reward_model.extract_trajectory_features)
    baseline_idx = int(prediction["features"][:, -1].argmax())
    baseline_ade = _ade(prediction["candidates"][baseline_idx])

    print(f"Responsibility-aware selection ADE: {selected_ade:.3f} m")
    print(f"Highest-log-prob baseline ADE:      {baseline_ade:.3f} m")


if __name__ == "__main__":
    main()

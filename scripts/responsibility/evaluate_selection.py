"""Evaluates responsibility-aware trajectory selection (paper Sec. V-C) against
ablations, on paired candidates.

For every scenario, queried agent and prediction step P, it draws one set of
K candidate futures and lets every selector pick from that same set, so
differences between selectors are paired and not sampling noise:

  responsibility  level filtered causally from beta windows closed by P (Eq. 6)
                  -> that level's reward network (the paper's method)
  oracle_level    level from the HMM's smoothed posterior over the whole clip
                  (uses the future; an upper bound on knowing the level)
  random_level    a uniformly random level's reward network
  pooled_reward   one reward network trained on all levels (H = 1), if given
  most_likely     the candidate the model itself rates most probable
  random          a uniformly random candidate
  min_ade         the best candidate in hindsight (the selection ceiling)

The paper's Table I compares a learned level-conditioned reward to a hand-
tuned cost, changing two things at once; responsibility vs pooled_reward
isolates the level conditioning, and responsibility vs random_level
checks that the levels carry information at all.

Rows go to --out-csv as they are computed (re-running skips finished
scenarios); a summary with standard errors is printed at the end.

Example:
    python -m scripts.responsibility.evaluate_selection \
        --ckpt ckpts/clsft_E9.ckpt --data-root data/womd_cache/validation \
        --hmm logs/responsibility/hmm.pkl --reward-nets logs/responsibility/reward_nets.pt \
        --pooled-reward-net logs/responsibility/reward_pooled.pt \
        --query sdc --out-csv logs/responsibility/eval.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.responsibility._cli import (  # noqa: E402
    add_model_args,
    add_responsibility_args,
    agent_type_from_args,
    responsibility_config_from_args,
)
from src.responsibility.dataset_builder import (  # noqa: E402
    SCOPE_NEIGHBOURS,
    compute_scenario_responsibility,
    window_steps,
)
from src.responsibility.interaction import select_query_agents  # noqa: E402
from src.responsibility.metrics import ground_truth_boxes  # noqa: E402
from src.responsibility.motion_set import MotionSetSynthesizer, shifted_context  # noqa: E402
from src.responsibility.pipeline import (  # noqa: E402
    compose_responsibility_config,
    count_scenarios,
    default_sampling_scheme,
    iter_scenarios,
    load_hmm,
    load_reward_nets,
    load_smart_model,
    resolve_device,
)
from src.responsibility.selection import sample_candidates  # noqa: E402

SELECTORS = ["responsibility", "oracle_level", "random_level", "pooled_reward",
             "most_likely", "random", "min_ade"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--hmm", required=True)
    p.add_argument("--reward-nets", required=True)
    p.add_argument("--pooled-reward-net", default=None, help="From train_reward.py --pool-levels.")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--predict-steps", type=int, nargs="+", default=[40, 50, 60, 70],
                   help="Context steps to predict from within each clip.")
    p.add_argument("--n-candidates", type=int, default=25)
    p.add_argument("--first-scenario", type=int, default=0,
                   help="Dataset index to start from; skip the scenarios the HMM and rewards were fitted on.")
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--max-agents-per-scenario", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    add_model_args(p)
    add_responsibility_args(p)
    return p.parse_args()


def _nearest_level(level, available):
    return level if level in available else min(available, key=lambda a: abs(a - level))


def _errors(traj, gt, valid):
    err = torch.linalg.norm(traj - gt, dim=-1)  # [K, T]
    last = int((torch.arange(valid.numel(), device=valid.device) * valid).argmax())
    ade = (err * valid).sum(-1) / valid.sum().clamp(min=1)
    return ade, err[:, last]


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    device = resolve_device(args.device)
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=device)
    num_hist = hydra_cfg.model.model_config.decoder.num_historical_steps
    cfg = responsibility_config_from_args(args, num_hist)
    if cfg.others_future == "logged":
        print("WARNING: --others-future logged scores candidates against the others' real "
              "futures; that leaks the answer into prediction.", flush=True)
    synth = MotionSetSynthesizer(
        model, default_sampling_scheme(num_k=args.motion_set_num_k, temp=args.motion_set_temp),
        rollout_batch_size=args.rollout_batch_size,
    )
    hmm = load_hmm(args.hmm)
    nets = load_reward_nets(args.reward_nets)
    pooled = load_reward_nets(args.pooled_reward_net)[0] if args.pooled_reward_net else None
    levels = sorted(nets)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {row["scenario_id"] for row in csv.DictReader(f)}
    fields = ["scenario_id", "agent_id", "is_sdc", "predict_step", "n_history", "level", "oracle_level"]
    fields += [f"{s}_{m}" for s in SELECTORS for m in ("ade", "fde")]
    new_file = not out.exists()
    handle = open(out, "a", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if new_file:
        writer.writeheader()

    n_scen = 0
    indices = range(args.first_scenario, count_scenarios(args.data_root))
    for _, scenario_id, data in iter_scenarios(args.data_root, device, indices=indices):
        if args.max_scenarios is not None and n_scen >= args.max_scenarios:
            break
        n_scen += 1
        if scenario_id in done:
            continue
        now = num_hist - 1
        agents = select_query_agents(data, now, cfg.query_mode, agent_type_from_args(args))
        agents = agents[: args.max_agents_per_scenario] if args.max_agents_per_scenario else agents
        if not agents:
            continue

        # beta over every window of the clip; causal slices of it feed the
        # filter, the whole sequence feeds the smoothed (oracle) level
        steps = window_steps(data, cfg, synth, cfg.window_stride)
        obs = compute_scenario_responsibility(synth, data, cfg, steps=steps, agents=agents)
        by_agent = {}
        for o in obs:
            if o.scope == SCOPE_NEIGHBOURS:
                by_agent.setdefault(o.agent_idx, []).append(o)

        for a in agents:
            seq = sorted(by_agent.get(a, []), key=lambda o: o.k)
            if not seq:
                continue
            betas = np.stack([o.beta.numpy() for o in seq])
            smoothed = hmm.smooth(betas)
            for predict_step in args.predict_steps:
                causal = [i for i, o in enumerate(seq) if o.k + cfg.metric_horizon <= predict_step]
                if not causal or predict_step > steps[-1] or not bool(data["agent"]["valid_mask"][a, predict_step]):
                    continue
                last = causal[-1]
                level = _nearest_level(int(hmm.forward_filter(betas[: last + 1])[-1].argmax()), levels)
                oracle = _nearest_level(int(smoothed[last].argmax()), levels)

                with shifted_context(data, predict_step, num_hist) as shifted:
                    cand = sample_candidates(synth, shifted, a, cfg, args.n_candidates)
                    gt = ground_truth_boxes(shifted, [a], num_hist, cand.horizon)
                    gt_pos, gt_valid = gt.pos[0].clone(), gt.valid[0].clone()
                if not bool(gt_valid.any()):
                    continue
                ade, fde = _errors(cand.traj, gt_pos, gt_valid.to(cand.traj.dtype))

                pick = {
                    "responsibility": int(nets[level].reward(cand.features).argmax()),
                    "oracle_level": int(nets[oracle].reward(cand.features).argmax()),
                    "random_level": int(nets[int(rng.choice(levels))].reward(cand.features).argmax()),
                    "pooled_reward": int(pooled.reward(cand.features).argmax()) if pooled else None,
                    "most_likely": int(cand.log_prob.argmax()),
                    "random": int(rng.integers(len(cand.traj))),
                    "min_ade": int(ade.argmin()),
                }
                row = {
                    "scenario_id": scenario_id,
                    "agent_id": int(data["agent"]["id"][a]),
                    "is_sdc": int(bool(data["agent"]["role"][a, 0])),
                    "predict_step": predict_step,
                    "n_history": len(causal),
                    "level": level,
                    "oracle_level": oracle,
                }
                for name, idx in pick.items():
                    row[f"{name}_ade"] = "" if idx is None else f"{float(ade[idx]):.4f}"
                    row[f"{name}_fde"] = "" if idx is None else f"{float(fde[idx]):.4f}"
                writer.writerow(row)
        handle.flush()
        print(f"{n_scen}: {scenario_id} done", flush=True)
    handle.close()
    summarize(out)


def summarize(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("no rows")
        return
    print(f"\n{len(rows)} predictions from {len({r['scenario_id'] for r in rows})} scenarios")
    print(f"{'selector':16s} {'ADE (m)':>16s} {'FDE (m)':>16s}")
    for s in SELECTORS:
        cells = []
        for m in ("ade", "fde"):
            v = np.array([float(r[f"{s}_{m}"]) for r in rows if r[f"{s}_{m}"] != ""])
            cells.append(f"{v.mean():7.3f} +- {v.std(ddof=1) / math.sqrt(len(v)):5.3f}" if len(v) > 1 else "n/a")
        print(f"{s:16s} {cells[0]:>16s} {cells[1]:>16s}")
    paired = np.array([float(r["responsibility_ade"]) - float(r["most_likely_ade"]) for r in rows])
    print(f"\nresponsibility - most_likely ADE (paired): {paired.mean():+.3f} +- "
          f"{paired.std(ddof=1) / math.sqrt(len(paired)) if len(paired) > 1 else float('nan'):.3f} m")
    if rows[0]["pooled_reward_ade"] != "":
        paired = np.array([float(r["responsibility_ade"]) - float(r["pooled_reward_ade"]) for r in rows])
        print(f"responsibility - pooled_reward ADE (paired): {paired.mean():+.3f} +- "
              f"{paired.std(ddof=1) / math.sqrt(len(paired)) if len(paired) > 1 else float('nan'):.3f} m")


if __name__ == "__main__":
    main()

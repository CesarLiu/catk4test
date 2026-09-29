"""Stage 3: train per-responsibility-level reward networks (paper Sec. IV-B /
V-C) from the level-labeled observations produced by fit_hmm.py.

Walks the scenario cache once, regenerating auxiliary trajectories and
features for the scenarios the labeled observations refer to, one scenario
at a time. Pass the same responsibility flags as stage 1: the features are
measured over --metric-horizon under --neighbor-future.

Example:
    python -m scripts.responsibility.train_reward \
        --ckpt ckpts/clsft_E9.ckpt \
        --data-root data/womd_cache/validation \
        --labeled-observations logs/responsibility/observations_labeled.pkl \
        --out logs/responsibility/reward_nets.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.responsibility._cli import (  # noqa: E402
    add_model_args,
    add_responsibility_args,
    responsibility_config_from_args,
)
from src.responsibility.pipeline import (  # noqa: E402
    compose_responsibility_config,
    default_sampling_scheme,
    iter_scenarios,
    load_observations,
    load_smart_model,
    resolve_device,
    save_reward_nets,
    train_reward_stage,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--labeled-observations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-aux", type=int, default=24, help="Auxiliary (negative) samples per observation.")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    add_model_args(p)
    add_responsibility_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    observations = load_observations(args.labeled_observations)
    needed = {o.scenario_id for o in observations if o.level is not None}
    print(f"{len(observations)} observations across {len(needed)} scenarios need features.")

    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    device = resolve_device(args.device)
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=device)
    rcfg = responsibility_config_from_args(args, hydra_cfg.model.model_config.decoder.num_historical_steps)

    scenarios = (
        (scenario_id, data)
        for _, scenario_id, data in iter_scenarios(args.data_root, device)
        if scenario_id in needed
    )
    bundles = train_reward_stage(
        model,
        observations,
        scenarios,
        rcfg,
        n_aux=args.n_aux,
        sampling_scheme=default_sampling_scheme(num_k=args.motion_set_num_k, temp=args.motion_set_temp),
        hidden_dim=args.hidden_dim,
        n_epochs=args.n_epochs,
        lr=args.lr,
        rollout_batch_size=args.rollout_batch_size,
    )
    save_reward_nets(bundles, args.out)
    print(f"Trained reward networks for levels {sorted(bundles.keys())}. Saved to {args.out}")


if __name__ == "__main__":
    main()

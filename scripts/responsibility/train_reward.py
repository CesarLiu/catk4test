"""Stage 3: train per-responsibility-level reward networks (paper Sec. IV-B /
V-C) from the level-labeled observations produced by fit_hmm.py.

Re-reads the scenario cache (restricted to the scenarios referenced by the
labeled observations) to regenerate auxiliary trajectories and features,
since stage 1 does not persist full scenario tensors by default.

Example:
    python -m scripts.responsibility.train_reward \
        --ckpt /path/to/smart.ckpt \
        --data-root /path/to/cache/validation \
        --labeled-observations logs/responsibility/observations_labeled.pkl \
        --out logs/responsibility/reward_nets.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.responsibility.metrics import ResponsibilityConfig  # noqa: E402
from src.responsibility.pipeline import (  # noqa: E402
    build_dataloader,
    compose_responsibility_config,
    default_sampling_scheme,
    load_observations,
    load_smart_model,
    save_reward_nets,
    train_reward_stage,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--labeled-observations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-config", default="smart")
    p.add_argument("--n-aux", type=int, default=24)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--collision-weight", type=float, default=10.0)
    p.add_argument("--drivable-half-width", type=float, default=3.0)
    p.add_argument("--motion-set-num-k", type=int, default=2048)
    p.add_argument("--motion-set-temp", type=float, default=1.0)
    p.add_argument("--metric-horizon", type=int, default=20)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    observations = load_observations(args.labeled_observations)
    needed_scenarios = {o.scenario_id for o in observations if o.level is not None}
    print(f"{len(observations)} observations across {len(needed_scenarios)} scenarios need features.")

    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=args.device)

    dataloader = build_dataloader(args.data_root)
    data_by_scenario = {}
    for data in dataloader:
        scenario_id = str(data["scenario_id"][0]) if "scenario_id" in data else None
        if scenario_id in needed_scenarios:
            data_by_scenario[scenario_id] = data
        if len(data_by_scenario) == len(needed_scenarios):
            break

    rcfg = ResponsibilityConfig(
        collision_weight=args.collision_weight,
        drivable_half_width=args.drivable_half_width,
        metric_horizon=args.metric_horizon,
        num_historical_steps=hydra_cfg.model.model_config.decoder.num_historical_steps,
    )
    sampling_scheme = default_sampling_scheme(num_k=args.motion_set_num_k, temp=args.motion_set_temp)

    bundles = train_reward_stage(
        model,
        observations,
        data_by_scenario,
        rcfg,
        n_aux=args.n_aux,
        sampling_scheme=sampling_scheme,
        hidden_dim=args.hidden_dim,
        n_epochs=args.n_epochs,
        lr=args.lr,
    )
    save_reward_nets(bundles, args.out)
    print(f"Trained reward networks for levels {sorted(bundles.keys())}. Saved to {args.out}")


if __name__ == "__main__":
    main()

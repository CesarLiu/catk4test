"""Stage 1: compute counterfactual safety/courtesy responsibility observations
over a directory of cached WOMD scenarios (paper Sec. III).

Example:
    python -m scripts.responsibility.compute_responsibility \
        --ckpt /path/to/smart.ckpt \
        --data-root /path/to/cache/validation \
        --out logs/responsibility/observations.pkl \
        --max-scenarios 200
"""

import argparse
import sys
from pathlib import Path

# allow running as `python scripts/responsibility/compute_responsibility.py`
# from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.responsibility.metrics import ResponsibilityConfig  # noqa: E402
from src.responsibility.pipeline import (  # noqa: E402
    build_dataloader,
    compose_responsibility_config,
    compute_dataset_responsibility,
    default_sampling_scheme,
    load_smart_model,
    save_observations,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="Path to a SMART .ckpt checkpoint.")
    p.add_argument("--data-root", required=True, help="Directory of cached scenario .pkl files.")
    p.add_argument("--out", required=True, help="Output .pkl path for the observations.")
    p.add_argument("--model-config", default="smart", help="configs/model/<name>.yaml group.")
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--max-agents-per-scenario", type=int, default=None)
    p.add_argument("--agent-type", type=int, default=0, help="0=veh,1=ped,2=cyc,-1=all")
    p.add_argument("--n-safety-samples", type=int, default=40)
    p.add_argument("--cvar-alpha", type=float, default=0.1)
    p.add_argument("--n-courtesy-samples", type=int, default=20)
    p.add_argument("--neighbor-radius", type=float, default=30.0)
    p.add_argument("--d-sat", type=float, default=10.0, help="D_g saturation range in m; <=0 disables.")
    p.add_argument("--metric-horizon", type=int, default=20, help="Future steps @10Hz used to measure beta.")
    p.add_argument("--window-stride", type=int, default=5,
                   help="Spacing between context steps swept within each clip.")
    p.add_argument("--no-sweep-windows", action="store_true",
                   help="Evaluate only the clip's native current step (sequences of length 1).")
    p.add_argument("--courtesy-skip-distance", type=float, default=20.0, help="<=0 disables the skip.")
    p.add_argument("--collision-weight", type=float, default=10.0)
    p.add_argument("--drivable-half-width", type=float, default=3.0)
    p.add_argument("--motion-set-num-k", type=int, default=2048)
    p.add_argument("--motion-set-temp", type=float, default=1.0)
    p.add_argument("--rollout-batch-size", type=int, default=8,
                   help="Scene copies per inference call; raise for speed, lower for GPU memory.")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=args.device)

    dataloader = build_dataloader(args.data_root)
    rcfg = ResponsibilityConfig(
        n_safety_samples=args.n_safety_samples,
        cvar_alpha=args.cvar_alpha,
        n_courtesy_samples=args.n_courtesy_samples,
        neighbor_radius=args.neighbor_radius,
        d_sat=args.d_sat if args.d_sat > 0 else None,
        metric_horizon=args.metric_horizon,
        window_stride=args.window_stride,
        courtesy_skip_distance=(
            args.courtesy_skip_distance if args.courtesy_skip_distance > 0 else None
        ),
        num_historical_steps=hydra_cfg.model.model_config.decoder.num_historical_steps,
        collision_weight=args.collision_weight,
        drivable_half_width=args.drivable_half_width,
    )
    sampling_scheme = default_sampling_scheme(num_k=args.motion_set_num_k, temp=args.motion_set_temp)
    agent_type_filter = None if args.agent_type < 0 else args.agent_type

    observations, _ = compute_dataset_responsibility(
        model,
        dataloader,
        rcfg,
        sampling_scheme=sampling_scheme,
        agent_type_filter=agent_type_filter,
        max_agents_per_scenario=args.max_agents_per_scenario,
        max_scenarios=args.max_scenarios,
        sweep_windows=not args.no_sweep_windows,
        rollout_batch_size=args.rollout_batch_size,
        keep_data=False,
    )
    save_observations(observations, args.out)
    print(f"Saved {len(observations)} responsibility observations to {args.out}")


if __name__ == "__main__":
    main()

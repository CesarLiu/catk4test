"""Stage 1: compute counterfactual safety/courtesy responsibility observations
over a directory of cached WOMD scenarios (paper Sec. III).

Example:
    python -m scripts.responsibility.compute_responsibility \
        --ckpt ckpts/clsft_E9.ckpt \
        --data-root data/womd_cache/validation \
        --out logs/responsibility/observations.pkl \
        --query sdc --direction both --max-scenarios 200
"""

import argparse
import sys
from pathlib import Path

# allow running as `python scripts/responsibility/compute_responsibility.py`
# from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.responsibility._cli import (  # noqa: E402
    add_model_args,
    add_responsibility_args,
    agent_type_from_args,
    responsibility_config_from_args,
)
from src.responsibility.pipeline import (  # noqa: E402
    compose_responsibility_config,
    compute_dataset_responsibility,
    default_sampling_scheme,
    iter_scenarios,
    load_smart_model,
    resolve_device,
    save_observations,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="Directory of cached scenario .pkl files.")
    p.add_argument("--out", required=True, help="Output .pkl path for the observations.")
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--max-agents-per-scenario", type=int, default=None)
    p.add_argument("--no-sweep-windows", action="store_true",
                   help="Evaluate only the clip's native current step (sequences of length 1).")
    add_model_args(p)
    add_responsibility_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    device = resolve_device(args.device)
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=device)
    rcfg = responsibility_config_from_args(args, hydra_cfg.model.model_config.decoder.num_historical_steps)

    observations, _ = compute_dataset_responsibility(
        model,
        iter_scenarios(args.data_root, device),
        rcfg,
        sampling_scheme=default_sampling_scheme(num_k=args.motion_set_num_k, temp=args.motion_set_temp),
        agent_type_filter=agent_type_from_args(args),
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

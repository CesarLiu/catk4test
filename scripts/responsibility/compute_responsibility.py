"""Stage 1: compute counterfactual safety/courtesy responsibility observations
over a directory of cached WOMD scenarios (paper Sec. III).

Two output modes:

  --out-dir DIR   one result file per scenario under DIR/obs/, resumable
                  (finished scenarios are skipped on restart) and shardable
                  (--num-shards N --shard-index i, several processes may fill
                  the same DIR). DIR/config.json records the run's settings
                  and a restart with different ones is refused. Use this for
                  anything long.
  --out FILE.pkl  everything in one file at the end; for quick runs.

Later stages accept either a file or a results directory.

Example:
    python -m scripts.responsibility.compute_responsibility \
        --ckpt ckpts/clsft_E9.ckpt \
        --data-root data/womd_cache/validation \
        --out-dir logs/responsibility/run_sdc \
        --query sdc --direction both --courtesy-estimator exact
"""

import argparse
import sys
from dataclasses import asdict
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
    run_responsibility_to_dir,
    save_observations,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="Directory of cached scenario .pkl files.")
    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument("--out-dir", help="Resumable results directory (one file per scenario).")
    out.add_argument("--out", help="Single output .pkl, written at the end.")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--allow-config-mismatch", action="store_true",
                   help="Add to an --out-dir whose recorded configuration differs.")
    p.add_argument("--max-scenarios", type=int, default=None, help="Per shard.")
    p.add_argument("--max-agents-per-scenario", type=int, default=None)
    p.add_argument("--no-sweep-windows", action="store_true",
                   help="Evaluate only the clip's native current step (sequences of length 1).")
    add_model_args(p)
    add_responsibility_args(p)
    args = p.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        p.error("--shard-index must be in [0, --num-shards)")
    return args


def main():
    args = parse_args()
    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    device = resolve_device(args.device)
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=device)
    rcfg = responsibility_config_from_args(args, hydra_cfg.model.model_config.decoder.num_historical_steps)
    sampling_scheme = default_sampling_scheme(num_k=args.motion_set_num_k, temp=args.motion_set_temp)
    common = dict(
        sampling_scheme=sampling_scheme,
        agent_type_filter=agent_type_from_args(args),
        max_agents_per_scenario=args.max_agents_per_scenario,
        max_scenarios=args.max_scenarios,
        sweep_windows=not args.no_sweep_windows,
        rollout_batch_size=args.rollout_batch_size,
    )

    if args.out_dir:
        # everything that changes the numbers; shard layout and batch size do not
        run_config = {
            "responsibility": asdict(rcfg),
            "ckpt": str(Path(args.ckpt).resolve()),
            "model_config": args.model_config,
            "data_root": str(Path(args.data_root).resolve()),
            "sampling": {"num_k": args.motion_set_num_k, "temp": args.motion_set_temp},
            "agent_type": args.agent_type,
            "max_agents_per_scenario": args.max_agents_per_scenario,
            "sweep_windows": not args.no_sweep_windows,
        }
        totals = run_responsibility_to_dir(
            model, args.data_root, args.out_dir, rcfg, run_config, device=device,
            num_shards=args.num_shards, shard_index=args.shard_index,
            allow_config_mismatch=args.allow_config_mismatch, **common,
        )
        print(f"shard {args.shard_index}/{args.num_shards} done: {totals}")
        return

    observations, _ = compute_dataset_responsibility(
        model, iter_scenarios(args.data_root, device), rcfg, keep_data=False, **common
    )
    save_observations(observations, args.out)
    print(f"Saved {len(observations)} responsibility observations to {args.out}")


if __name__ == "__main__":
    main()

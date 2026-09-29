"""Measures counterfactual rollout throughput against --rollout-batch-size on
the current machine, to pick the batch size for stage 1.

Each measurement runs the same set of single-agent rollout jobs (one scene
copy each, cycling over the observed vehicles) through the packed engine at
one batch size, and reports copies per second and peak GPU memory. Memory
grows with the number of agents in a scene, so benchmark on a busy scenario
(--scenario-index) and leave headroom for larger ones.

Example:
    python -m scripts.responsibility.benchmark_rollouts \
        --ckpt ckpts/clsft_E9.ckpt --data-root data/womd_cache/validation \
        --batch-sizes 16 32 64 128 256 512 1024
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.responsibility.motion_set import MotionSetSynthesizer, RolloutJob  # noqa: E402
from src.responsibility.pipeline import (  # noqa: E402
    compose_responsibility_config,
    count_scenarios,
    default_sampling_scheme,
    iter_scenarios,
    load_smart_model,
    resolve_device,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--model-config", default="smart")
    p.add_argument("--scenario-index", type=int, default=None,
                   help="Default: the scenario with the most agents among the first --scan.")
    p.add_argument("--scan", type=int, default=50)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512])
    p.add_argument("--copies", type=int, default=512, help="Scene copies per measurement.")
    p.add_argument("--horizon", type=int, default=20, help="Rollout steps @10Hz.")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=device)
    now = hydra_cfg.model.model_config.decoder.num_historical_steps - 1

    if args.scenario_index is None:
        n = min(args.scan, count_scenarios(args.data_root))
        sizes = [(d["agent"]["position"].shape[0], i) for i, _, d in iter_scenarios(args.data_root, "cpu", range(n))]
        _, index = max(sizes)
    else:
        index = args.scenario_index
    (_, scenario_id, data), = iter_scenarios(args.data_root, device, [index])
    n_agent = data["agent"]["position"].shape[0]
    n_pl = data["map_save"]["traj_pos"].shape[0]
    vehicles = ((data["agent"]["type"] == 0) & data["agent"]["valid_mask"][:, now]).nonzero(as_tuple=True)[0].tolist()
    print(f"device {device}"
          + (f" ({torch.cuda.get_device_name()})" if device.startswith("cuda") else "")
          + f"; scenario {scenario_id} (index {index}): {n_agent} agents, {n_pl} map polylines, "
          f"{len(vehicles)} observed vehicles; {args.copies} copies of {args.horizon} steps per measurement")

    scheme = default_sampling_scheme()
    jobs = [RolloutJob(free=(vehicles[i % len(vehicles)],), n_samples=1) for i in range(args.copies)]

    rows = []
    for batch in args.batch_sizes:
        synth = MotionSetSynthesizer(model, scheme, rollout_batch_size=batch)
        tok_map, tok_agent = synth.tokenize(data)
        try:
            synth.run_jobs(tok_map, tok_agent, jobs[:batch], rollout_horizon=args.horizon)  # warm-up
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            _sync(device)
            t0 = time.perf_counter()
            synth.run_jobs(tok_map, tok_agent, jobs, rollout_horizon=args.horizon)
            _sync(device)
            dt = time.perf_counter() - t0
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {batch:5d}: out of memory")
            torch.cuda.empty_cache()
            break
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else float("nan")
        rows.append((batch, args.copies / dt, peak))
        print(f"  batch {batch:5d}: {args.copies / dt:8.1f} copies/s   peak memory {peak:6.1f} GiB", flush=True)

    if rows:
        best = max(rows, key=lambda r: r[1])
        print(f"\nfastest: --rollout-batch-size {best[0]} ({best[1]:.1f} copies/s, {best[2]:.1f} GiB on "
              f"a {n_agent}-agent scene). Memory scales with agents per scene, so keep headroom.")


if __name__ == "__main__":
    main()

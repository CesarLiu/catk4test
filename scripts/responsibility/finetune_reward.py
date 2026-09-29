"""Stage 5: fine-tune SMART with the per-level responsibility rewards
(paper Sec. IV-B; method in src/responsibility/finetune.py).

Needs the level-labeled observations of stage 2 and the reward networks of
stage 3 for the *training* scenarios; only scenarios with labeled agents are
used, and only labeled agents are rewarded (all agents are simulated and
enter the KL term). Optionally evaluates on held-out scenarios every
--eval-every steps (KL, reward, collision rate, ADE / minADE of the rollouts
against the log), starting with the untouched model at step 0.

Writes to --out-dir:
  last.ckpt, step_XXXXXX.ckpt   fine-tuned weights in the base checkpoint's
                                format: pass them as --ckpt to any script
  train_state.pt                optimizer and progress, for --resume
  train_log.csv, eval_log.csv   per-step training and evaluation statistics
  config.json                   the settings of the run

Example:
    python -m scripts.responsibility.finetune_reward \
        --ckpt ckpts/clsft_E9.ckpt --data-root data/womd_cache/training \
        --labeled-observations logs/responsibility/run_train/labeled.pkl \
        --reward-nets logs/responsibility/run_train/reward_nets.pt \
        --val-data-root data/womd_cache/validation \
        --val-labeled-observations logs/responsibility/run_val/labeled.pkl \
        --out-dir logs/responsibility/finetune --steps 2000
"""

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.responsibility._cli import (  # noqa: E402
    add_model_args,
    add_responsibility_args,
    responsibility_config_from_args,
)
from src.responsibility.finetune import (  # noqa: E402
    OTHERS_MODES,
    REWARD_TRANSFORMS,
    FinetuneConfig,
    RewardFinetuner,
    frozen_copy,
    level_reward_fn,
    level_table,
    save_finetuned_checkpoint,
)
from src.responsibility.pipeline import (  # noqa: E402
    compose_responsibility_config,
    iter_scenarios,
    load_observations,
    load_reward_nets,
    load_smart_model,
    resolve_device,
    scenario_files,
)
from src.smart.utils.finetune import set_model_for_finetuning  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="Training scenarios (cached .pkl directory).")
    p.add_argument("--labeled-observations", required=True, help="Stage-2 output for the training scenarios.")
    p.add_argument("--reward-nets", required=True, help="Stage-3 output.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-data-root", default=None)
    p.add_argument("--val-labeled-observations", default=None,
                   help="Levels of the held-out scenarios; without them only KL and ADE are reported.")
    p.add_argument("--val-scenarios", type=int, default=50)

    g = p.add_argument_group("optimisation")
    g.add_argument("--steps", type=int, default=2000, help="Optimizer steps.")
    g.add_argument("--scenes-per-step", type=int, default=4, help="Scenes whose losses are accumulated per step.")
    g.add_argument("--lr", type=float, default=1e-5)
    g.add_argument("--warmup-steps", type=int, default=50)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--max-grad-norm", type=float, default=1.0)
    g.add_argument("--train-all", action="store_true",
                   help="Train every parameter; default: CAT-K's fine-tuning set (agent attention layers "
                        "and token head; map encoder and embeddings frozen).")

    g = p.add_argument_group("objective")
    d = FinetuneConfig()
    g.add_argument("--n-rollouts", type=int, default=d.n_rollouts, help="Rollouts per scene (the advantage group).")
    g.add_argument("--horizon", type=int, default=None, help="10Hz steps rolled out; default --metric-horizon.")
    g.add_argument("--kl-coef", type=float, default=d.kl_coef)
    g.add_argument("--reward-transform", default=d.reward_transform, choices=REWARD_TRANSFORMS)
    g.add_argument("--others", default=d.others, choices=OTHERS_MODES,
                   help="Collision feature against the others' rollouts in the same scene (rollout) "
                        "or against --others-future (config, as in stage 3).")
    g.add_argument("--no-normalize-advantage", action="store_true")
    g.add_argument("--level-max-gap", type=int, default=10,
                   help="Use an agent's level only from windows starting within this many steps of the "
                        "clip's current step.")

    g = p.add_argument_group("bookkeeping")
    g.add_argument("--eval-every", type=int, default=200)
    g.add_argument("--save-every", type=int, default=500)
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--resume", action="store_true", help="Continue from --out-dir/train_state.pt.")
    add_model_args(p)
    add_responsibility_args(p)
    return p.parse_args()


def _index_by_id(data_root):
    return {f.stem: i for i, f in enumerate(scenario_files(data_root))}


def _append_csv(path: Path, row: dict):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def training_scenes(data_root, indices, device, seed):
    """Endless stream of (scenario_id, data), reshuffled every epoch."""
    rng = random.Random(seed)
    order = list(indices)
    while True:
        rng.shuffle(order)
        for _, scenario_id, data in iter_scenarios(data_root, device, indices=order):
            yield scenario_id, data


def evaluate(tuner, val_root, val_indices, val_levels, device):
    totals, counts = {}, {}
    for _, scenario_id, data in iter_scenarios(val_root, device, indices=val_indices):
        for key, value in tuner.evaluate(data, val_levels.get(scenario_id, {})).items():
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    device = resolve_device(args.device)
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=device)
    num_hist = hydra_cfg.model.model_config.decoder.num_historical_steps
    rcfg = responsibility_config_from_args(args, num_hist)
    cfg = FinetuneConfig(
        n_rollouts=args.n_rollouts,
        horizon=args.horizon or rcfg.metric_horizon,
        kl_coef=args.kl_coef,
        reward_transform=args.reward_transform,
        others=args.others,
        normalize_advantage=not args.no_normalize_advantage,
    )

    # the reference is the untouched base model, also when resuming
    ref_model = frozen_copy(model)
    set_model_for_finetuning(model.encoder, not args.train_all)
    if args.train_all:
        for p in model.parameters():
            p.requires_grad_(True)
    model.eval()  # dropout off; gradients still flow
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"training {sum(p.numel() for p in params):,} of {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps))
    )

    reward_nets = load_reward_nets(args.reward_nets)
    tuner = RewardFinetuner(model, level_reward_fn(reward_nets, rcfg, cfg), cfg, rcfg, ref_model=ref_model)

    levels = level_table(load_observations(args.labeled_observations), num_hist - 1, args.level_max_gap)
    by_id = _index_by_id(args.data_root)
    train_indices = sorted(by_id[s] for s in levels if s in by_id)
    n_agents = sum(len(v) for s, v in levels.items() if s in by_id)
    if not train_indices:
        raise SystemExit("no labeled scenario of --labeled-observations is in --data-root")
    print(f"{len(train_indices)} training scenarios with {n_agents} rewarded agents; "
          f"levels {sorted({z for v in levels.values() for z in v.values()})}; reward nets {sorted(reward_nets)}")

    val_indices, val_levels = [], {}
    if args.val_data_root:
        if args.val_labeled_observations:
            val_levels = level_table(load_observations(args.val_labeled_observations), num_hist - 1, args.level_max_gap)
            val_by_id = _index_by_id(args.val_data_root)
            val_indices = sorted(val_by_id[s] for s in val_levels if s in val_by_id)[: args.val_scenarios]
        else:
            val_indices = list(range(min(args.val_scenarios, len(scenario_files(args.val_data_root)))))

    run_config = {"args": vars(args), "finetune": asdict(cfg), "responsibility": asdict(rcfg)}
    (out / "config.json").write_text(json.dumps(run_config, indent=2, default=str))

    step = 0
    state_path = out / "train_state.pt"
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        torch.set_rng_state(state["torch_rng"])
        print(f"resumed at step {step}")

    def save(tag):
        extra = {"step": step, "finetune": asdict(cfg), "base_checkpoint": str(args.ckpt)}
        save_finetuned_checkpoint(model, args.ckpt, str(out / f"{tag}.ckpt"), extra)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step,
                    "torch_rng": torch.get_rng_state()}, state_path)

    def run_eval():
        if not val_indices:
            return
        stats = evaluate(tuner, args.val_data_root, val_indices, val_levels, device)
        _append_csv(out / "eval_log.csv", {"step": step, **{k: f"{v:.5f}" for k, v in sorted(stats.items())}})
        print(f"[eval step {step}] " + "  ".join(f"{k} {v:.4f}" for k, v in sorted(stats.items())), flush=True)

    if step == 0:
        run_eval()

    scenes = training_scenes(args.data_root, train_indices, device, args.seed + step)
    started = time.time()
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        agg = {}
        for _ in range(args.scenes_per_step):
            scenario_id, data = next(scenes)
            loss, stats = tuner.loss(data, levels[scenario_id])
            (loss / args.scenes_per_step).backward()
            for k, v in stats.items():
                agg.setdefault(k, []).append(v)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.log_every == 0 or step == 1:
            row = {"step": step, "lr": f"{scheduler.get_last_lr()[0]:.3g}", "grad_norm": f"{float(grad_norm):.4f}"}
            row.update({k: f"{float(np.mean(v)):.5f}" for k, v in sorted(agg.items())})
            _append_csv(out / "train_log.csv", row)
            rate = step / max(time.time() - started, 1e-9)
            print(f"[step {step}/{args.steps}] " + "  ".join(f"{k} {v}" for k, v in row.items() if k != "step")
                  + f"  ({rate * 60:.1f} steps/min)", flush=True)
        if step % args.eval_every == 0:
            run_eval()
        if step % args.save_every == 0 or step == args.steps:
            save(f"step_{step:06d}")
            save("last")
    print(f"done; fine-tuned checkpoint: {out / 'last.ckpt'}")


if __name__ == "__main__":
    main()

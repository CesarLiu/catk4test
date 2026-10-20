"""Flags scenes by their single worst stretch of responsibility: each agent's
beta_s and beta_c sequences (from compute_responsibility.py) are smoothed
over time, a sliding window of --window seconds finds the most dangerous
(largest mean beta_s) and the least courteous (largest mean beta_c)
stretch, and the scene is flagged if that stretch crosses the threshold --
one dangerous cut-in is enough, however calm the rest of the clip is.

With --channel either (the default) a scene is flagged when beta_s reaches
--safety-threshold or beta_c reaches --courtesy-threshold; the score is
the larger of the two in units of its threshold.

Given hand labels (--positives), it also reports how well the score ranks
them (AUROC, average precision, precision at the number of positives) and
the precision / recall of the flags; --sweep tries every combination of
--sweep-windows and --sweep-widths instead.

Examples:
    python -m scripts.responsibility.flag_scenes \\
        --observations logs/responsibility/run_adv500 --window 1.5 \\
        --out-csv logs/responsibility/run_adv500/flags.csv
    python -m scripts.responsibility.flag_scenes \\
        --observations logs/responsibility/run_adv500 --sweep \\
        --positives 0 12 55 62 68 75 86 136 149 176 210 227 234 242 250 283 304 \\
                    313 319 343 366 376 378 384 387 398 418 430 464 478 488
"""

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from src.responsibility.dataset_builder import SCOPE_NEIGHBOURS, group_sequences  # noqa: E402
from src.responsibility.results_io import load_observations  # noqa: E402
from src.responsibility.scene_flags import (  # noqa: E402
    CHANNELS,
    SMOOTHERS,
    FlagConfig,
    ranking_metrics,
    score_sequences,
)


def parse_args():
    d = FlagConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--observations", required=True, help="compute_responsibility.py --out-dir, or a .pkl.")
    p.add_argument("--smoother", default=d.smoother, choices=SMOOTHERS,
                   help="Temporal smoothing before the sliding window; ema is causal.")
    p.add_argument("--smooth-width", type=float, default=d.smooth_width,
                   help="s: Gaussian sigma, moving-average width or EMA time constant.")
    p.add_argument("--window", type=float, default=d.window,
                   help="s: length of the sliding window whose mean beta is scored (at least one beta window).")
    p.add_argument("--channel", default=d.channel, choices=CHANNELS)
    p.add_argument("--safety-threshold", type=float, default=d.safety_threshold, help="m")
    p.add_argument("--courtesy-threshold", type=float, default=d.courtesy_threshold, help="nats")
    p.add_argument("--positives", nargs="+", default=None, help="Scenario ids labelled positive (e.g. aggressive).")
    p.add_argument("--exclude", nargs="+", default=[], help="Scenario ids left out of the evaluation.")
    p.add_argument("--sweep", action="store_true", help="Evaluate every --sweep-windows x --sweep-widths (needs --positives).")
    p.add_argument("--sweep-windows", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--sweep-widths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    p.add_argument("--out-csv", default=None, help="One row per scene and agent, worst first.")
    args = p.parse_args()
    if args.sweep and not args.positives:
        p.error("--sweep needs --positives")
    return args


def evaluate(scores, positives, exclude, cfg):
    keys = [k for k in scores if k[0] not in exclude]
    y = [int(k[0] in positives) for k in keys]
    return {
        channel: ranking_metrics([scores[k][channel] for k in keys], y,
                                 flagged=[scores[k]["flagged"] for k in keys] if channel == cfg.channel else None)
        for channel in CHANNELS
    }


def fmt(m):
    p = m["positives"]
    return f"AUROC {m.get('auroc', float('nan')):.3f}  AP {m.get('ap', float('nan')):.3f}  " \
           f"P@{p} {m.get(f'precision@{p}', float('nan')):.3f}"


def main():
    args = parse_args()
    cfg = FlagConfig(smoother=args.smoother, smooth_width=args.smooth_width, window=args.window,
                     channel=args.channel, safety_threshold=args.safety_threshold,
                     courtesy_threshold=args.courtesy_threshold)
    observations = [o for o in load_observations(args.observations) if o.scope == SCOPE_NEIGHBOURS]
    grouped = group_sequences(observations)
    print(f"{len(grouped)} sequences from {len(observations)} observations")
    positives, exclude = set(args.positives or []), set(args.exclude)

    if args.sweep:
        print(f"smoother {args.smoother}; window (s) x smooth width (s); AUROC of safety / courtesy / either")
        best = None
        for window in args.sweep_windows:
            for width in args.sweep_widths:
                c = replace(cfg, window=window, smooth_width=width)
                m = evaluate(score_sequences(grouped, c), positives, exclude, c)
                print(f"  window {window:4.1f}  width {width:4.2f}:  " + "  ".join(
                    f"{ch} {m[ch].get('auroc', float('nan')):.3f}" for ch in CHANNELS))
                if best is None or m[cfg.channel]["auroc"] > best[0]:
                    best = (m[cfg.channel]["auroc"], window, width)
        print(f"best {cfg.channel}: AUROC {best[0]:.3f} at window {best[1]} s, smooth width {best[2]} s")
        return

    scores = score_sequences(grouped, cfg)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1]["score"])
    flagged = [k for k, s in ranked if s["flagged"]]
    print(f"{len(flagged)} of {len(scores)} flagged ({cfg.channel}, safety >= {cfg.safety_threshold} m, "
          f"courtesy >= {cfg.courtesy_threshold} nats, window {cfg.window} s, {cfg.smoother} {cfg.smooth_width} s)")
    for name in ("safety", "courtesy"):
        v = np.array([s[name] for s in scores.values()])
        print(f"  worst-window {name}: " + ", ".join(f"p{q} {np.percentile(v, q):.3f}" for q in (50, 90, 95, 99)))
    if positives:
        m = evaluate(scores, positives, exclude, cfg)
        for channel in CHANNELS:
            print(f"  {channel:8s} {fmt(m[channel])}")
        f = m[cfg.channel]
        print(f"  flags: {f['flagged']} flagged, precision {f['precision']:.3f}, recall {f['recall']:.3f}")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scenario_id", "agent_id", "flagged", "score", "safety", "safety_t0", "safety_t1",
                        "courtesy", "courtesy_t0", "courtesy_t1", "n_windows"]
                       + (["label"] if positives else []))
            for (sid, aid, _), s in ranked:
                w.writerow([sid, aid, int(s["flagged"]), f"{s['score']:.4f}", f"{s['safety']:.4f}", *s["safety_t"],
                            f"{s['courtesy']:.4f}", *s["courtesy_t"], s["n_windows"]]
                           + ([int(sid in positives)] if positives else []))
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()

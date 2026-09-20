"""Reproduces the paper's offline ex-post analysis plots (Sec. V-B, Fig. 2-3):
the safety-vs-courtesy scatter of raw responsibility observations colored by
HMM level, and the BIC-vs-number-of-levels curve.

Example:
    python -m scripts.responsibility.ex_post_analysis \
        --labeled-observations logs/responsibility/observations_labeled.pkl \
        --bic-summary logs/responsibility/hmm_summary.json \
        --out-dir logs/responsibility/plots
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.responsibility.pipeline import load_observations  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labeled-observations", required=True)
    p.add_argument("--bic-summary", default=None)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def plot_scatter(observations, out_path: Path):
    betas = np.stack([o.beta.numpy() for o in observations])
    levels = np.array([o.level if o.level is not None else -1 for o in observations])

    fig, ax = plt.subplots(figsize=(6, 5))
    n_levels = levels.max() + 1
    cmap = plt.get_cmap("viridis", max(n_levels, 1))
    for level in sorted(set(levels.tolist())):
        mask = levels == level
        label = f"z={level}" if level >= 0 else "unlabeled"
        color = cmap(level) if level >= 0 else "gray"
        ax.scatter(betas[mask, 0], betas[mask, 1], s=8, alpha=0.6, label=label, color=color)
    ax.set_xlabel(r"$\beta^{\mathrm{safety}}$")
    ax.set_ylabel(r"$\beta^{\mathrm{courtesy}}$")
    ax.set_title("Responsibility observations by HMM level")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bic(summary: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(summary["n_states_grid"], summary["bic_by_n_states"], "o-")
    ax.axvline(summary["best_n_states"], color="r", linestyle="--", label=f"selected H={summary['best_n_states']}")
    ax.set_xlabel("number of responsibility levels H")
    ax.set_ylabel("BIC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    observations = load_observations(args.labeled_observations)
    plot_scatter(observations, out_dir / "responsibility_scatter.png")
    print(f"Wrote {out_dir / 'responsibility_scatter.png'}")

    if args.bic_summary is not None:
        with open(args.bic_summary) as f:
            summary = json.load(f)
        plot_bic(summary, out_dir / "bic_vs_levels.png")
        print(f"Wrote {out_dir / 'bic_vs_levels.png'}")

    level_counts = {}
    for o in observations:
        if o.level is not None:
            level_counts[o.level] = level_counts.get(o.level, 0) + 1
    print("Observations per level:", dict(sorted(level_counts.items())))


if __name__ == "__main__":
    main()

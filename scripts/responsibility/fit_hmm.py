"""Stage 2: fit the Gaussian HMM responsibility-level abstraction (paper
Sec. IV-A) on the observations produced by compute_responsibility.py, with
BIC-based model selection over the number of levels, and label every
observation with its most-likely level (Eq. 6).

Example:
    python -m scripts.responsibility.fit_hmm \
        --observations logs/responsibility/observations.pkl \
        --out-hmm logs/responsibility/hmm.pkl \
        --out-labeled logs/responsibility/observations_labeled.pkl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.responsibility.pipeline import (  # noqa: E402
    fit_hmm_stage,
    label_observations_with_levels,
    load_observations,
    save_hmm,
    save_observations,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", required=True)
    p.add_argument("--out-hmm", required=True)
    p.add_argument("--out-labeled", required=True)
    p.add_argument("--out-summary", default=None, help="Optional JSON path for the BIC sweep (for ex_post_analysis.py).")
    p.add_argument("--n-states-grid", type=int, nargs="+", default=[3, 5, 7, 9, 11])
    p.add_argument("--n-iter", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    observations = load_observations(args.observations)
    print(f"Loaded {len(observations)} observations.")

    result = fit_hmm_stage(
        observations,
        n_states_grid=args.n_states_grid,
        n_iter=args.n_iter,
        seed=args.seed,
        verbose=args.verbose,
    )
    for h, bic in zip(result.n_states_grid, result.bic_by_n_states):
        marker = " <- selected" if h == result.best_n_states else ""
        print(f"H={h}: BIC={bic:.2f}{marker}")

    label_observations_with_levels(observations, result.best_model)
    save_hmm(result.best_model, args.out_hmm)
    save_observations(observations, args.out_labeled)
    print(f"Selected H={result.best_n_states}. Saved HMM to {args.out_hmm}")
    print(f"Saved labeled observations to {args.out_labeled}")

    if args.out_summary is not None:
        Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_summary, "w") as f:
            json.dump(
                {
                    "n_states_grid": result.n_states_grid,
                    "bic_by_n_states": result.bic_by_n_states,
                    "best_n_states": result.best_n_states,
                },
                f,
                indent=2,
            )
        print(f"Saved BIC summary to {args.out_summary}")


if __name__ == "__main__":
    main()

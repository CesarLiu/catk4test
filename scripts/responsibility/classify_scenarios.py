"""Tags interacting pairs in cached WOMD scenarios with their interaction
type, scene type and interaction window (``src/responsibility/scenario_tags.py``),
optionally joined with stage-1 responsibility observations.

Reads the cached ``.pkl`` scenarios directly (no model; without
--observations not even torch_geometric), in the same order as every other
script here, so ``--max-scenarios 500``
covers the same 500 scenarios as ``compute_responsibility --max-scenarios 500``.
With ``--observations`` only the scenarios that have observations are tagged,
whatever ``--max-scenarios`` says.

Pairs (``--pairs``):
  auto          the annotated interactive pair (WOMD objects of interest, as in
                validation_interactive) where exactly two are marked, else the
                SDC and its tightest partner
  interest      only the annotated pair
  sdc           the SDC and the agent it interacts with most tightly
  all           every interacting pair, tightest first (--max-pairs per scenario)
  observations  every (queried agent, neighbour) pair with a non-zero
                responsibility in --observations

With --observations, each row also gets the responsibility of a toward b
over the windows that overlap the interaction window: how many, the mean and
max safety / courtesy, and (with labeled observations from fit_hmm) the most
common level.

Outputs: --out-csv, one row per pair; --out-summary (JSON), counts per
interaction type, scene type and their cross table, and window durations.

Example:
    python -m scripts.responsibility.classify_scenarios \\
        --data-root data/womd_cache/validation --max-scenarios 500 \\
        --observations logs/responsibility/observations_labeled.pkl --pairs auto \\
        --out-csv logs/responsibility/scenario_tags.csv \\
        --out-summary logs/responsibility/scenario_tags_summary.json
"""

import argparse
import csv
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.responsibility.results_io import load_observations, scenario_files  # noqa: E402
from src.responsibility.scenario_tags import (  # noqa: E402
    PAIR_MODES,
    TagConfig,
    interacting_pairs,
    tag_pair,
)

PAIRS_OBSERVATIONS = "observations"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="Cached scenario .pkl directory.")
    p.add_argument("--max-scenarios", type=int, default=500)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--observations", default=None,
                   help="Stage-1 observations (.pkl or results directory), optionally labeled by fit_hmm.")
    p.add_argument("--pairs", default="auto", choices=list(PAIR_MODES) + [PAIRS_OBSERVATIONS])
    p.add_argument("--max-pairs", type=int, default=10, help="Per scenario, for --pairs all / observations.")
    p.add_argument("--metric-horizon", type=int, default=20,
                   help="Steps an observation at step k covers ([k+1, k+H]); as in compute_responsibility.")
    p.add_argument("--scene-radius", type=float, default=TagConfig.scene_radius)
    p.add_argument("--time-gap", type=float, default=TagConfig.time_gap,
                   help="s: local PET / time headway under which a pair is interacting.")
    p.add_argument("--proximity-gap", type=float, default=TagConfig.proximity_gap)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-summary", default=None)
    return p.parse_args()


def _scenario_id(data, path: Path) -> str:
    raw = data.get("scenario_id", path.stem)
    return str(raw[0]) if isinstance(raw, (list, tuple)) else str(raw)


def _index_observations(observations):
    by_scenario = defaultdict(list)
    for obs in observations:
        by_scenario[str(obs.scenario_id)].append(obs)
    return by_scenario


def _observation_pairs(data, observations, max_pairs):
    ids = data["agent"]["id"].tolist()
    index = {int(i): n for n, i in enumerate(ids)}
    best = {}
    for obs in observations:
        for nbr_id, values in (obs.per_neighbour or {}).items():
            safety, courtesy = float(values[0]), float(values[1])
            if safety == 0.0 and courtesy == 0.0:
                continue
            key = (int(obs.agent_id), int(nbr_id))
            best[key] = max(best.get(key, (0.0, 0.0)), (safety, courtesy))
    ranked = sorted(best, key=lambda k: best[k], reverse=True)
    pairs = []
    for a_id, b_id in ranked:
        if a_id in index and b_id in index:
            pairs.append((index[a_id], index[b_id], PAIRS_OBSERVATIONS))
        if len(pairs) >= max_pairs:
            break
    return pairs


def _responsibility_in_window(observations, id_a, id_b, start, end, horizon):
    """a's responsibility toward b over the observations whose metric window
    [k+1, k+horizon] overlaps the interaction window [start, end]."""
    rows = []
    for obs in observations:
        if int(obs.agent_id) != id_a or start < 0:
            continue
        if obs.k + horizon < start or obs.k + 1 > end:
            continue
        values = (obs.per_neighbour or {}).get(id_b)
        if values is None:
            continue
        rows.append((float(values[0]), float(values[1]), obs.level))
    out = {"n_obs": len(rows), "safety_mean": math.nan, "safety_max": math.nan,
           "courtesy_mean": math.nan, "courtesy_max": math.nan, "level_mode": ""}
    if rows:
        s = torch.tensor([r[0] for r in rows])
        c = torch.tensor([r[1] for r in rows])
        out.update(safety_mean=float(s.mean()), safety_max=float(s.max()),
                   courtesy_mean=float(c.mean()), courtesy_max=float(c.max()))
        levels = [r[2] for r in rows if r[2] is not None]
        if levels:
            out["level_mode"] = Counter(levels).most_common(1)[0][0]
    return out


def summarize(rows, n_scenarios, n_without_pairs):
    types = Counter(r["interaction_type"] for r in rows)
    scenes = Counter(r["scene_type"] for r in rows)
    cross = defaultdict(Counter)
    for r in rows:
        cross[r["scene_type"]][r["interaction_type"]] += 1
    durations = defaultdict(list)
    for r in rows:
        if r["start_step"] >= 0:
            durations[r["interaction_type"]].append(r["duration_s"])
    return {
        "n_scenarios": n_scenarios,
        "n_scenarios_without_pairs": n_without_pairs,
        "n_pairs": len(rows),
        "interaction_type": dict(types.most_common()),
        "scene_type": dict(scenes.most_common()),
        "scene_x_interaction": {k: dict(v.most_common()) for k, v in cross.items()},
        "window_duration_s": {
            k: {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "n": len(v)} for k, v in durations.items()
        },
    }


def main():
    args = parse_args()
    cfg = TagConfig(scene_radius=args.scene_radius, time_gap=args.time_gap, proximity_gap=args.proximity_gap)
    files = scenario_files(args.data_root)
    obs_by_scenario = None
    if args.observations:
        obs_by_scenario = _index_observations(load_observations(args.observations))
    elif args.pairs == PAIRS_OBSERVATIONS:
        raise SystemExit("--pairs observations needs --observations")

    if obs_by_scenario is not None:
        # exactly the scenarios responsibility was computed on
        files = [f for f in files if f.stem in obs_by_scenario]
        if not files:
            raise SystemExit("none of the observations' scenario ids match a file in --data-root")
    else:
        files = files[args.start_index : args.start_index + args.max_scenarios]

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, n_without = [], 0
    writer = None
    with open(out, "w", newline="") as fh:
        for n, path in enumerate(files):
            with open(path, "rb") as f:
                data = pickle.load(f)
            sid = _scenario_id(data, path)
            observations = obs_by_scenario.get(sid, []) if obs_by_scenario is not None else []
            if args.pairs == PAIRS_OBSERVATIONS:
                pairs = _observation_pairs(data, observations, args.max_pairs)
            else:
                pairs = interacting_pairs(data, args.pairs, cfg, max_pairs=args.max_pairs)
            if not pairs:
                n_without += 1
            for a, b, source in pairs:
                row = tag_pair(data, a, b, scenario_id=sid, pair_source=source, cfg=cfg).as_row()
                if obs_by_scenario is not None:
                    row.update(_responsibility_in_window(
                        observations, row["id_a"], row["id_b"], row["start_step"], row["end_step"],
                        args.metric_horizon,
                    ))
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                fh.flush()
                rows.append(row)
            if (n + 1) % 50 == 0:
                print(f"{n + 1}/{len(files)} scenarios, {len(rows)} pairs", flush=True)

    summary = summarize(rows, len(files), n_without)
    print(json.dumps({k: summary[k] for k in ("n_scenarios", "n_scenarios_without_pairs", "n_pairs",
                                               "interaction_type", "scene_type")}, indent=2))
    if args.out_summary:
        Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_summary).write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()

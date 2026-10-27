"""Report figures: does responsibility agree with things a safety engineer
already trusts? Three checks, each run when its inputs are given:

  control  Known answer. The same car in the same scene, once as logged and
           once driving CAT's attack trajectory (benchmark_advgen.py CSVs in
           ../cat): how well does safety responsibility tell them apart?
  physics  Physical risk, measured on the log, per responsibility level: how
           often a nearby car had to brake hard, how often the queried car
           came within a short time-to-collision of someone, how close it got.
  rules    Rule baseline. The same windows judged by simple rules (TTC, time
           headway, hard braking or acceleration of the car itself);
           agreement at window and scene level, and the scenes where the two
           disagree, as a shortlist for videos.

Reads a run of either implementation:
  --catk-obs     this repo's stage-1 results directory or .pkl; labelled
                 observations (fit_hmm.py --out-labeled) carry HMM levels
  --cat-run      ../cat compute_responsibility.py output (windows*.csv),
                 optionally with --cat-levels (fit_levels.py levels.csv)
  --scenes       the scenario files that run was computed on (catk cache
                 *.pkl named by scenario id, or CAT's raw_scenes_500)
  --advgen       ../cat benchmark_advgen.py CSVs (the control check only)

Needs only numpy, matplotlib and torch (for catk results), not SMART: catk
observations are read without importing the model code.

Writes to OUT: control.png, physics.png, rules.png, cases.csv, summary.json
and summary.md (the headline numbers and the case shortlist).

Example (SMART run on the server, plus CAT's attacker benchmark):
    python -m scripts.responsibility.report_figures \\
        --catk-obs logs/responsibility/run_sdc/labeled.pkl \\
        --scenes data/womd_cache/validation \\
        --advgen ../cat/logs/advgen/part_*.csv --out-dir logs/responsibility/report
"""

import argparse
import csv
import glob
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ORANGE, BLUE, GREY, INK = "#C2410C", "#1F5FA8", "#8A909C", "#16213A"
DT = 0.1  # s per step (WOMD, 10 Hz)


# ---------------------------------------------------------------- loading

class _Observation:
    """Stand-in for catk's ResponsibilityObservation, so its pickles load
    without importing SMART."""


class _ObservationUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("src.responsibility") and name == "ResponsibilityObservation":
            return _Observation
        return super().find_class(module, name)


def _catk_observation_files(path: Path) -> List[Path]:
    if path.is_dir():
        folder = path / "obs" if (path / "obs").is_dir() else path
        return sorted(folder.glob("*.pkl"))
    return [path]


def load_catk_windows(path: str) -> List[Dict]:
    """One row per (agent, window) from catk observations, the max over all
    interacting neighbours (scope "neighbours") only."""
    rows = []
    for f in _catk_observation_files(Path(path)):
        with open(f, "rb") as handle:
            observations = _ObservationUnpickler(handle).load()
        for o in observations:
            if getattr(o, "scope", "neighbours") != "neighbours":
                continue
            beta = np.asarray(o.beta, dtype=float).reshape(-1)
            level = getattr(o, "level", None)
            rows.append({
                "scene": str(o.scenario_id), "scenario_id": str(o.scenario_id), "agent": str(int(o.agent_id)),
                "agent_index": int(o.agent_idx), "step": int(o.k),
                "safety": float(beta[0]), "courtesy": float(beta[1]),
                "level": -1 if level is None else int(level), "aggressive": None,
            })
    return rows


def load_cat_windows(run_dir: str, levels_csv: Optional[str] = None, levels_run: Optional[str] = None) -> List[Dict]:
    """Rows of ../cat windows*.csv (a re-run scene's repeated rows dropped),
    joined with the levels and verdicts of fit_levels.py when given."""
    rows = {}
    for f in sorted(Path(run_dir).glob("windows*.csv")):
        with open(f, newline="") as handle:
            for r in csv.DictReader(handle):
                key = (r["scene"], r["agent_id"], int(r["step"]))
                rows[key] = {
                    "scene": r["scene"], "scenario_id": r["scenario_id"], "agent": r["agent_id"],
                    "agent_index": None, "step": int(r["step"]),
                    "safety": float(r["safety"]), "courtesy": float(r["courtesy"]),
                    "level": -1, "aggressive": None,
                }
    if levels_csv:
        run = levels_run or Path(run_dir).name
        with open(levels_csv, newline="") as handle:
            levels = [r for r in csv.DictReader(handle)]
        if any(r.get("run") == run for r in levels):
            levels = [r for r in levels if r.get("run") == run]
        for r in levels:
            key = (r["scene"], r["agent_id"], int(r["step"]))
            if key in rows:
                rows[key]["level"] = int(r["level"])
                rows[key]["aggressive"] = int(r["aggressive"])
    return list(rows.values())


class Tracks:
    """Every agent of one scenario as arrays over the clip (10 Hz)."""

    def __init__(self, ids, kind, pos, vel, heading, length, width, valid):
        self.ids = [str(i) for i in ids]
        self.kind = np.asarray(kind)  # "vehicle", "pedestrian", "cyclist", "other"
        self.pos = np.asarray(pos, dtype=float)[..., :2]  # [N, T, 2]
        self.vel = np.asarray(vel, dtype=float)  # [N, T, 2]
        self.heading = np.asarray(heading, dtype=float)  # [N, T]
        self.length = np.asarray(length, dtype=float)  # [N]
        self.width = np.asarray(width, dtype=float)  # [N]
        self.valid = np.asarray(valid, dtype=bool)  # [N, T]

    @property
    def n_steps(self) -> int:
        return self.pos.shape[1]

    def index(self, agent_id: str) -> int:
        return self.ids.index(str(agent_id))

    @classmethod
    def from_file(cls, path: Path) -> "Tracks":
        with open(path, "rb") as f:
            data = pickle.load(f)
        if "tracks" in data:  # MetaDrive scenario description (../cat raw_scenes_500)
            tracks = data["tracks"]
            ids = list(tracks)
            states = [tracks[k]["state"] for k in ids]
            kinds = {"VEHICLE": "vehicle", "PEDESTRIAN": "pedestrian", "CYCLIST": "cyclist"}

            def per_agent(key):
                out = []
                for s in states:
                    v, ok = np.asarray(s[key], dtype=float), np.asarray(s["valid"], dtype=bool)
                    out.append(np.median(v[ok]) if ok.any() else np.nan)
                return np.array(out)

            return cls(ids, [kinds.get(tracks[k]["type"], "other") for k in ids],
                       np.stack([s["position"] for s in states]), np.stack([s["velocity"] for s in states]),
                       np.stack([s["heading"] for s in states]), per_agent("length"), per_agent("width"),
                       np.stack([s["valid"] for s in states]))
        agent = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in data["agent"].items()}  # catk cache
        kinds = np.array(["vehicle", "pedestrian", "cyclist"] + ["other"] * 253)[agent["type"].astype(int)]
        return cls(agent["id"].astype(int), kinds, agent["position"], agent["velocity"], agent["heading"],
                   agent["shape"][:, 0], agent["shape"][:, 1], agent["valid_mask"])


def scene_path(scenes: Path, scene: str) -> Path:
    return scenes / f"{scene}.pkl"


# ---------------------------------------------------------------- physics

def speed_and_accel(tr: Tracks) -> (np.ndarray, np.ndarray):
    """Speed [N, T] and longitudinal acceleration [N, T] (central difference
    over +-0.2 s; NaN where the agent is not valid at both ends)."""
    speed = np.linalg.norm(tr.vel, axis=-1)
    speed = np.where(tr.valid, speed, np.nan)
    acc = np.full_like(speed, np.nan)
    acc[:, 2:-2] = (speed[:, 4:] - speed[:, :-4]) / (4 * DT)
    return speed, acc


def circles(tr: Tracks, agents: np.ndarray, steps: np.ndarray) -> (np.ndarray, np.ndarray):
    """Three-circle footprint: centres [A, S, 3, 2] and radii [A]."""
    length = np.nan_to_num(tr.length[agents], nan=4.5)
    width = np.nan_to_num(tr.width[agents], nan=2.0)
    offset = np.maximum(length - width, 0.0) / 2.0  # [A]
    h = tr.heading[np.ix_(agents, steps)]  # [A, S]
    direction = np.stack([np.cos(h), np.sin(h)], axis=-1)  # [A, S, 2]
    k = np.array([-1.0, 0.0, 1.0])
    centres = tr.pos[np.ix_(agents, steps)][:, :, None, :] + (offset[:, None, None, None] * k[None, None, :, None]
                                                              * direction[:, :, None, :])
    return centres, width / 2.0


def time_to_collision(d: np.ndarray, v: np.ndarray, r: np.ndarray, cap: float) -> np.ndarray:
    """Constant-velocity time until two discs touch: relative position d
    [..., 2], relative velocity v [..., 2], radius sum r [...]. 0 if they
    already overlap, ``cap`` if they never touch (or are separating)."""
    a = (v * v).sum(-1)
    b = 2.0 * (d * v).sum(-1)
    c = (d * d).sum(-1) - r * r
    disc = b * b - 4.0 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
    hit = (b < 0) & (disc >= 0) & (a > 1e-9)
    out = np.where(hit, np.clip(t, 0.0, cap), cap)
    return np.where(c <= 0, 0.0, out)


def window_physics(tr: Tracks, speed: np.ndarray, acc: np.ndarray, agent: int, step: int, horizon: int,
                   cfg) -> Optional[Dict]:
    """Physical risk of ``agent`` over the window (step, step + horizon]."""
    steps = np.arange(step + 1, min(step + horizon, tr.n_steps - 1) + 1)
    steps = steps[tr.valid[agent, steps]]
    if steps.size == 0:
        return None
    others = np.array([j for j in range(len(tr.ids)) if j != agent and tr.valid[j, steps].any()])
    near = []
    if others.size:
        start = steps[0]
        ok = tr.valid[others, start]
        dist0 = np.linalg.norm(tr.pos[others, start] - tr.pos[agent, start], axis=-1)
        others = others[~ok | (dist0 < cfg.prefilter)]
    out = {"speed": float(np.nanmean(speed[agent, steps])), "min_gap": cfg.far, "min_ttc": cfg.ttc_cap,
           "min_thw": np.inf, "own_brake": float(np.nanmax(np.r_[-acc[agent, steps], 0.0])),
           "own_accel": float(np.nanmax(np.r_[acc[agent, steps], 0.0])), "near_brake": 0.0}
    if others.size == 0:
        return out
    ca, ra = circles(tr, np.array([agent]), steps)  # [1, S, 3, 2]
    cb, rb = circles(tr, others, steps)  # [B, S, 3, 2]
    both = tr.valid[others][:, steps] & tr.valid[agent, steps][None]  # [B, S]
    d = cb[:, :, None, :, :] - ca[:, :, :, None, :]  # [B, S, 3a, 3b, 2]
    r = (ra[0] + rb)[:, None, None, None]
    gap = np.linalg.norm(d, axis=-1) - r  # [B, S, 3, 3]
    gap = np.where(both[:, :, None, None], gap, np.inf).min(axis=(2, 3))  # [B, S]
    v = (tr.vel[others][:, steps] - tr.vel[agent, steps][None])[:, :, None, None, :]
    ttc = time_to_collision(d, np.broadcast_to(v, d.shape), np.broadcast_to(r, d.shape[:-1]), cfg.ttc_cap)
    ttc = np.where(both[:, :, None, None], ttc, cfg.ttc_cap).min(axis=(2, 3))  # [B, S]
    out["min_gap"] = float(max(min(gap.min(), cfg.far), 0.0))
    out["min_ttc"] = float(ttc.min())

    # time headway to a vehicle ahead in roughly the same lane and direction
    h = tr.heading[agent, steps]
    rel = tr.pos[others][:, steps] - tr.pos[agent, steps][None]  # [B, S, 2]
    lon = rel[..., 0] * np.cos(h) + rel[..., 1] * np.sin(h)
    lat = -rel[..., 0] * np.sin(h) + rel[..., 1] * np.cos(h)
    dh = np.abs((tr.heading[others][:, steps] - h[None] + np.pi) % (2 * np.pi) - np.pi)
    lengths = np.nan_to_num(tr.length, nan=4.5)
    bumper = lon - (lengths[agent] + lengths[others][:, None]) / 2.0
    ahead = both & (lon > 0) & (np.abs(lat) < cfg.lane_half_width) & (dh < np.radians(30)) & (bumper > 0)
    v_a = speed[agent, steps][None].repeat(len(others), 0)
    ahead &= v_a > cfg.thw_min_speed
    if ahead.any():
        out["min_thw"] = float((bumper[ahead] / v_a[ahead]).min())

    # the hardest braking of a nearby vehicle (one that came within cfg.near m)
    nearby = others[(gap.min(axis=1) < cfg.near) & (tr.kind[others] == "vehicle")]
    if nearby.size:
        out["near_brake"] = float(np.nanmax(np.r_[-acc[np.ix_(nearby, steps)].ravel(), 0.0]))
    return out


def rule_reasons(p: Dict, cfg) -> List[str]:
    reasons = []
    if p["min_ttc"] < cfg.rule_ttc:
        reasons.append("ttc")
    if p["min_thw"] < cfg.rule_thw:
        reasons.append("headway")
    if max(p["own_brake"], p["own_accel"]) > cfg.rule_accel:
        reasons.append("hard_accel")
    return reasons


# ---------------------------------------------------------------- verdicts

def level_order(rows: List[Dict], calm: float = 0.5) -> (Dict[int, int], List[int]):
    """Ranks the HMM levels by how aggressive their windows are (mean safety
    and courtesy, each in units of its spread, summed: as ../cat levels.py)
    and marks those standing ``calm`` spreads above the calmest one in
    either. Returns old level -> rank, and the aggressive ranks."""
    x = np.array([[r["safety"], r["courtesy"]] for r in rows if r["level"] >= 0])
    z = np.array([r["level"] for r in rows if r["level"] >= 0])
    scale = np.maximum(x.std(axis=0), 1e-9)
    levels = sorted(set(z.tolist()))
    means = np.array([x[z == lv].mean(axis=0) for lv in levels])
    order = np.argsort((means / scale).sum(axis=1), kind="stable")
    rank = {levels[i]: r for r, i in enumerate(order)}
    base = means[order[0]]
    aggressive = [r for r, i in enumerate(order) if ((means[i] - base) / scale > calm).any()]
    return rank, aggressive


def label_windows(rows: List[Dict], judged: Sequence[Dict], quantile: float) -> str:
    """Fills ``aggressive`` (and ranks ``level``) on every row. Levels from
    an HMM when the run has them, else the threshold rule of ../cat
    summarize_responsibility.py: above the q-quantile of the positive
    values of this run. Returns which one was used."""
    if any(r["aggressive"] is not None for r in rows):
        return "levels"
    if any(r["level"] >= 0 for r in rows):
        rank, aggressive = level_order(rows)
        for r in rows:
            if r["level"] >= 0:
                r["level"] = rank[r["level"]]
                r["aggressive"] = int(r["level"] in aggressive)
        return "levels"
    thresholds = []
    for key in ("safety", "courtesy"):
        positive = np.array([r[key] for r in judged if r[key] > 0])
        thresholds.append(np.quantile(positive, quantile) if positive.size else np.inf)
    for r in rows:
        r["aggressive"] = int(r["safety"] > thresholds[0] or r["courtesy"] > thresholds[1])
    return "threshold"


# ---------------------------------------------------------------- statistics

def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(score of a random positive > a random negative), ties count half."""
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    greater = (positive[:, None] > negative[None]).mean()
    ties = (positive[:, None] == negative[None]).mean()
    return float(greater + 0.5 * ties)


def scene_bootstrap(values: np.ndarray, scenes: np.ndarray, stat, n: int = 1000, seed: int = 0) -> (float, float):
    """95% interval of ``stat`` over windows, resampling whole scenes."""
    rng = np.random.default_rng(seed)
    unique = np.unique(scenes)
    if unique.size < 2:
        return float("nan"), float("nan")
    groups = [values[scenes == s] for s in unique]
    draws = []
    for _ in range(n):
        pick = rng.integers(0, unique.size, unique.size)
        draws.append(stat(np.concatenate([groups[i] for i in pick])))
    return tuple(float(q) for q in np.nanpercentile(draws, [2.5, 97.5]))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    observed = (a == b).mean()
    expected = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


# ---------------------------------------------------------------- plots

def _style(plt):
    plt.rcParams.update({
        "font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18, "axes.edgecolor": GREY,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK, "ytick.color": INK,
        "axes.labelcolor": INK, "text.color": INK, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def plot_control(groups: Dict[str, np.ndarray], threshold: float, out: Path):
    import matplotlib.pyplot as plt

    _style(plt)
    names = list(groups)
    colors = {"Real drivers (logged)": BLUE, "CAT attacker": ORANGE}
    fig, ax = plt.subplots(figsize=(12, 1.6 + 1.5 * len(names)))
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        v = groups[name]
        y = i + rng.uniform(-0.18, 0.18, v.size)
        ax.scatter(v, y, s=36, alpha=0.6, color=colors.get(name, GREY), edgecolor="none")
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, lw=3)
        share = (v > threshold).mean() * 100
        ax.text(1.01, i, f"{share:.0f}% flagged", transform=ax.get_yaxis_transform(), va="center", fontsize=18,
                color=colors.get(name, INK), fontweight="bold")
    ax.axvline(threshold, color=INK, ls="--", lw=1.5)
    ax.text(threshold, len(names) - 0.45, "  alarm threshold", fontsize=15, color=INK, va="bottom")
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(-0.6, len(names) - 0.2)
    ax.set_xlabel("safety margin given up vs. the car's own alternatives (m)")
    fig.savefig(out)
    plt.close(fig)


def plot_physics(table: List[Dict], out: Path, with_ttc: bool = False):
    import matplotlib.pyplot as plt

    _style(plt)
    panels = [("near_brake_share", "a nearby car brakes hard", "% of windows", 100),
              ("median_gap", "closest gap to any road user", "median (m)", 1)]
    if with_ttc:
        panels.insert(1, ("ttc_share", "time-to-collision gets short", "% of windows", 100))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.2))
    x = np.arange(len(table))
    colors = [ORANGE if g["aggressive"] else BLUE for g in table]
    for ax, (key, title, unit, scale) in zip(axes, panels):
        values = np.array([g[key] for g in table]) * scale
        lo = np.array([g[key + "_ci"][0] for g in table]) * scale
        hi = np.array([g[key + "_ci"][1] for g in table]) * scale
        ax.bar(x, values, color=colors, width=0.7)
        ax.errorbar(x, values, yerr=[np.maximum(values - lo, 0), np.maximum(hi - values, 0)], fmt="none",
                    ecolor=INK, capsize=5, lw=1.5)
        ax.set_xticks(x, [g["label"] for g in table], rotation=0, fontsize=15)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(unit)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_rules(matrix: np.ndarray, out: Path):
    """2x2 scene counts: rows ML verdict (aggressive, normal), columns rule
    verdict (flagged, not flagged)."""
    import matplotlib.pyplot as plt

    _style(plt)
    fig, ax = plt.subplots(figsize=(9, 7))
    total = max(matrix.sum(), 1)
    fills = [["#F3D3C2", "#FBE9DE"], ["#D6E2F2", "#ECEAE3"]]
    notes = [["both say aggressive", "only responsibility"], ["only the rules", "both say normal"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=fills[i][j], ec="#FDFCF9", lw=6))
            ax.text(j + 0.5, 1.58 - i, f"{matrix[i, j]}", ha="center", va="center", fontsize=44, fontweight="bold")
            ax.text(j + 0.5, 1.30 - i, f"{100 * matrix[i, j] / total:.0f}% of scenes", ha="center", fontsize=16)
            ax.text(j + 0.5, 1.12 - i, notes[i][j], ha="center", fontsize=15, color="#4A5263")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["rules: flagged", "rules: not flagged"])
    ax.set_yticks([1.5, 0.5], ["responsibility:\naggressive", "responsibility:\nnormal"])
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------- checks

def control_check(paths: Sequence[str], fair_rule: Optional[str], out_dir: Path) -> Dict:
    rows = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            with open(f, newline="") as handle:
                rows += list(csv.DictReader(handle))

    def finite(values):
        v = np.array([float(x) for x in values if x not in ("", "nan", None)], dtype=float)
        return v[np.isfinite(v)]

    cat = [r for r in rows if r["rule"] == "cat"]
    logged = finite(r["logged_adversary_beta"] for r in cat)
    attacker = finite(r["beta"] for r in cat)
    groups = {"Real drivers (logged)": logged}
    if fair_rule:
        fair = finite(r["beta"] for r in rows if r["rule"] == fair_rule)
        if fair.size:
            groups[f"Fair attacker ({fair_rule})"] = fair
    groups["CAT attacker"] = attacker
    threshold = float(np.quantile(logged, 0.9))
    plot_control(groups, threshold, out_dir / "control.png")
    return {
        "scenes": int(len(cat)), "threshold_m": threshold, "auc_cat_vs_logged": auc(attacker, logged),
        "flagged": {k: float((v > threshold).mean()) for k, v in groups.items()},
        "median_m": {k: float(np.median(v)) for k, v in groups.items()},
    }


def physics_and_rules(rows: List[Dict], scenes: Path, cfg, out_dir: Path) -> Dict:
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)
    done = 0
    for scene, scene_rows in by_scene.items():
        path = scene_path(scenes, scene)
        if not path.exists():
            continue
        tr = Tracks.from_file(path)
        speed, acc = speed_and_accel(tr)
        for r in scene_rows:
            try:
                agent = r["agent_index"] if r["agent_index"] is not None else tr.index(r["agent"])
            except ValueError:
                continue
            p = window_physics(tr, speed, acc, agent, r["step"], cfg.horizon, cfg)
            if p is not None:
                r.update(p)
                r["rule_reasons"] = rule_reasons(p, cfg)
                r["agent_index"] = agent
        done += 1
    measured = [r for r in rows if "min_ttc" in r]
    judged = [r for r in measured if r["speed"] >= cfg.min_speed]
    if not judged:
        raise SystemExit(f"no window could be measured: are --scenes ({scenes}) the run's scenario files?")
    verdict = label_windows(rows, judged, cfg.quantile)

    # physics per level (or per verdict when there are no levels)
    s = np.array([r["scene"] for r in judged])
    for i, r in enumerate(judged):
        r["_i"] = i
    if verdict == "levels" and cfg.physics_by == "level":
        keys = sorted({r["level"] for r in judged})
        groups = [(f"level {k}", [r for r in judged if r["level"] == k]) for k in keys]
    else:
        groups = [("normal", [r for r in judged if not r["aggressive"]]),
                  ("aggressive", [r for r in judged if r["aggressive"]])]
    table = []
    for label, members in groups:
        if not members:
            continue
        idx = np.array([m["_i"] for m in members])
        brake = np.array([m["near_brake"] > cfg.hard_brake for m in members], dtype=float)
        ttc = np.array([m["min_ttc"] < cfg.risk_ttc for m in members], dtype=float)
        gap = np.array([m["min_gap"] for m in members])
        entry = {"label": label, "aggressive": bool(members[0]["aggressive"]), "windows": len(members),
                 "near_brake_share": float(brake.mean()), "ttc_share": float(ttc.mean()),
                 "median_gap": float(np.median(gap))}
        entry["near_brake_share_ci"] = scene_bootstrap(brake, s[idx], np.mean)
        entry["ttc_share_ci"] = scene_bootstrap(ttc, s[idx], np.mean)
        entry["median_gap_ci"] = scene_bootstrap(gap, s[idx], np.median)
        table.append(entry)
    plot_physics(table, out_dir / "physics.png", cfg.physics_ttc)

    pooled = {}
    for name, flag in (("aggressive", 1), ("normal", 0)):
        m = [r for r in judged if r["aggressive"] == flag]
        pooled[name] = {
            "windows": len(m),
            "near_brake_share": float(np.mean([r["near_brake"] > cfg.hard_brake for r in m])) if m else None,
            "ttc_share": float(np.mean([r["min_ttc"] < cfg.risk_ttc for r in m])) if m else None,
            "median_gap": float(np.median([r["min_gap"] for r in m])) if m else None,
        }

    # rules vs responsibility, per window and per scene
    ml_w = np.array([r["aggressive"] for r in judged], dtype=bool)
    rule_w = np.array([bool(r["rule_reasons"]) for r in judged])
    scenes_out = []
    per_scene = defaultdict(list)
    for r in judged:
        per_scene[(r["scene"], r["agent"])].append(r)
    for (scene, agent), w in per_scene.items():
        share = float(np.mean([x["aggressive"] for x in w]))
        ml = share >= cfg.scene_share if verdict == "levels" else share > 0
        reasons = sorted({x for r in w for x in r["rule_reasons"]})
        scenes_out.append({
            "scene": scene, "scenario_id": w[0]["scenario_id"], "agent": agent, "agent_index": w[0]["agent_index"],
            "windows": len(w), "aggressive_share": round(share, 3), "responsibility": "aggressive" if ml else "normal",
            "rules": "flagged" if reasons else "not flagged", "rule_reasons": "+".join(reasons),
            "max_safety_m": round(max(x["safety"] for x in w), 3),
            "max_courtesy_nats": round(max(x["courtesy"] for x in w), 3),
            "min_ttc_s": round(min(x["min_ttc"] for x in w), 2), "min_gap_m": round(min(x["min_gap"] for x in w), 2),
            "max_near_brake": round(max(x["near_brake"] for x in w), 2),
            "first_window": min(x["step"] for x in w),
        })
    ml_s = np.array([x["responsibility"] == "aggressive" for x in scenes_out])
    rule_s = np.array([x["rules"] == "flagged" for x in scenes_out])
    matrix = np.array([[np.sum(ml_s & rule_s), np.sum(ml_s & ~rule_s)],
                       [np.sum(~ml_s & rule_s), np.sum(~ml_s & ~rule_s)]])
    plot_rules(matrix, out_dir / "rules.png")
    for x in scenes_out:
        x["quadrant"] = {(True, True): "both", (True, False): "only_responsibility",
                         (False, True): "only_rules", (False, False): "neither"}[
            (x["responsibility"] == "aggressive", x["rules"] == "flagged")]
    reason_counts = defaultdict(int)
    for r in judged:
        for x in r["rule_reasons"]:
            reason_counts[x] += 1
    return {
        "verdict": verdict, "scenes_read": done, "windows_measured": len(measured), "windows_judged": len(judged),
        "aggressive_window_share": float(ml_w.mean()), "levels": table, "pooled": pooled,
        "rules": {
            "window_agreement": float((ml_w == rule_w).mean()), "window_kappa": cohen_kappa(ml_w, rule_w),
            "rule_window_share": float(rule_w.mean()), "reasons": dict(reason_counts),
            "scene_matrix": matrix.tolist(), "scene_kappa": cohen_kappa(ml_s, rule_s),
            "scene_agreement": float((ml_s == rule_s).mean()), "scenes": len(scenes_out),
        },
        "cases": scenes_out,
    }


def shortlist(cases: List[Dict], n: int) -> Dict[str, List[Dict]]:
    """The clearest examples per quadrant, for videos."""
    pick = {
        "both": lambda c: (-c["aggressive_share"], -c["max_courtesy_nats"]),
        "only_responsibility": lambda c: (-c["aggressive_share"], -c["max_courtesy_nats"]),
        "only_rules": lambda c: (c["min_ttc_s"], c["min_gap_m"]),
        "neither": lambda c: (-c["windows"], c["max_safety_m"]),
    }
    return {q: sorted([c for c in cases if c["quadrant"] == q], key=key)[:n] for q, key in pick.items()}


def video_command(case: Dict, args) -> str:
    if args.cat_run:
        return (f"python -m scripts.responsibility.visualize_responsibility --record {args.cat_run}/records/"
                f"{case['scene']}.pkl" + (f" --levels {args.cat_levels}" if args.cat_levels else "")
                + f" --out-dir logs/videos/case_{case['scene']}")
    files = sorted(p.stem for p in Path(args.scenes).glob("*.pkl"))
    index = files.index(case["scene"]) if case["scene"] in files else "?"
    return (f"python -m scripts.responsibility.visualize_motion_sets --ckpt $CKPT --data-root {args.scenes} "
            f"--scenario-index {index} --agent-index {case['agent_index']} --neighbor-future hidden "
            f"--out-dir logs/responsibility/video_{case['scene']}")


def write_markdown(summary: Dict, short: Dict, args, out: Path):
    lines = ["# Report numbers", ""]
    c = summary.get("control")
    if c:
        lines += ["## Known answer: attackers vs. real drivers", "",
                  f"- {c['scenes']} scenes; alarm threshold = 90th percentile of real drivers = {c['threshold_m']:.2f} m",
                  f"- AUC (CAT attacker vs. the same car as logged): {c['auc_cat_vs_logged']:.2f}"]
        lines += [f"- flagged: {k} {100 * v:.0f}% (median {c['median_m'][k]:+.2f} m)" for k, v in c["flagged"].items()]
        lines.append("")
    p = summary.get("physics")
    if p:
        a, n = p["pooled"]["aggressive"], p["pooled"]["normal"]
        lines += ["## Physical risk", "",
                  f"- verdicts from {p['verdict']}; {p['windows_judged']} judged windows "
                  f"({100 * p['aggressive_window_share']:.0f}% aggressive) in {p['scenes_read']} scenes", "",
                  "| group | windows | nearby car brakes hard | TTC short | median closest gap |",
                  "|---|---|---|---|---|"]
        for g in p["levels"]:
            lines.append(f"| {g['label']}{' (aggressive)' if g['aggressive'] else ''} | {g['windows']} | "
                         f"{100 * g['near_brake_share']:.0f}% | {100 * g['ttc_share']:.0f}% | {g['median_gap']:.1f} m |")
        if a["windows"] and n["windows"]:
            lines.append(f"| all aggressive | {a['windows']} | {100 * a['near_brake_share']:.0f}% | "
                         f"{100 * a['ttc_share']:.0f}% | {a['median_gap']:.1f} m |")
            lines.append(f"| all normal | {n['windows']} | {100 * n['near_brake_share']:.0f}% | "
                         f"{100 * n['ttc_share']:.0f}% | {n['median_gap']:.1f} m |")
        r = p["rules"]
        m = r["scene_matrix"]
        lines += ["", "## Rules vs. responsibility", "",
                  f"- rules: TTC < {args.rule_ttc} s, time headway < {args.rule_thw} s, |accel| > {args.rule_accel} m/s2; "
                  f"window reasons {r['reasons']}",
                  f"- windows: agreement {100 * r['window_agreement']:.0f}%, kappa {r['window_kappa']:.2f}",
                  f"- scenes ({r['scenes']}): both {m[0][0]}, only responsibility {m[0][1]}, only rules {m[1][0]}, "
                  f"neither {m[1][1]}; agreement {100 * r['scene_agreement']:.0f}%, kappa {r['scene_kappa']:.2f}",
                  "", "## Video shortlist", ""]
        for q, cases in short.items():
            lines.append(f"### {q}")
            for case in cases:
                lines.append(f"- scene {case['scene']} agent {case['agent']}: share {case['aggressive_share']}, "
                             f"max safety {case['max_safety_m']} m, max courtesy {case['max_courtesy_nats']} nats, "
                             f"min TTC {case['min_ttc_s']} s, rules: {case['rule_reasons'] or '-'}")
                lines.append(f"  `{video_command(case, args)}`")
            lines.append("")
    out.write_text("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--catk-obs", help="catk observations: results directory or .pkl (labelled: with levels).")
    src.add_argument("--cat-run", help="../cat compute_responsibility.py output directory.")
    p.add_argument("--cat-levels", help="../cat fit_levels.py levels.csv for --cat-run.")
    p.add_argument("--cat-levels-run", help="Run name in levels.csv (default: the --cat-run folder name).")
    p.add_argument("--scenes", help="Scenario files of the run (catk cache or CAT raw_scenes_500).")
    p.add_argument("--advgen", nargs="+", default=[], help="../cat benchmark_advgen.py CSVs (globs allowed).")
    p.add_argument("--fair-rule", default=None, help='Rule to show beside them, e.g. "fair@2,0.3".')
    p.add_argument("--out-dir", required=True)
    p.add_argument("--horizon", type=int, default=20, help="Steps per window (the run's metric horizon).")
    p.add_argument("--min-speed", type=float, default=1.0, help="Windows slower than this (m/s) are not judged.")
    p.add_argument("--quantile", type=float, default=0.9, help="Threshold verdicts when the run has no levels.")
    p.add_argument("--scene-share", type=float, default=0.5, help="Level verdicts: aggressive windows per scene.")
    p.add_argument("--rule-ttc", type=float, default=1.5)
    p.add_argument("--rule-thw", type=float, default=1.0, help="Time headway (s).")
    p.add_argument("--rule-accel", type=float, default=3.0, help="|longitudinal acceleration| (m/s2).")
    p.add_argument("--risk-ttc", type=float, default=2.0, help="Physics panel: 'TTC gets short'.")
    p.add_argument("--hard-brake", type=float, default=3.0, help="Physics panel: nearby braking (m/s2).")
    p.add_argument("--near", type=float, default=15.0, help="Nearby = came within this gap (m).")
    p.add_argument("--physics-by", default="verdict", choices=["verdict", "level"],
                   help="Physics bars: aggressive vs. normal, or one bar per HMM level.")
    p.add_argument("--physics-ttc", action="store_true", help="Add the short-TTC panel to physics.png.")
    p.add_argument("--n-cases", type=int, default=5)
    args = p.parse_args()
    if (args.catk_obs or args.cat_run) and not args.scenes:
        p.error("--scenes is required with --catk-obs or --cat-run")
    if not (args.catk_obs or args.cat_run or args.advgen):
        p.error("nothing to do: give a run (--catk-obs or --cat-run) and/or --advgen")
    args.prefilter, args.far, args.ttc_cap, args.lane_half_width, args.thw_min_speed = 50.0, 50.0, 10.0, 1.8, 2.0
    return args


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, short = {}, {}
    if args.advgen:
        summary["control"] = control_check(args.advgen, args.fair_rule, out)
        print(f"control: AUC {summary['control']['auc_cat_vs_logged']:.2f}, flagged {summary['control']['flagged']}")
    if args.catk_obs or args.cat_run:
        rows = (load_catk_windows(args.catk_obs) if args.catk_obs
                else load_cat_windows(args.cat_run, args.cat_levels, args.cat_levels_run))
        print(f"{len(rows)} windows")
        result = physics_and_rules(rows, Path(args.scenes), args, out)
        cases = result.pop("cases")
        short = shortlist(cases, args.n_cases)
        summary["physics"] = result
        with open(out / "cases.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(sorted(cases, key=lambda c: (c["quadrant"], -c["aggressive_share"])))
        print(f"physics: {result['pooled']}")
        print(f"rules: scene matrix {result['rules']['scene_matrix']}, kappa {result['rules']['scene_kappa']:.2f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_markdown(summary, short, args, out / "summary.md")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

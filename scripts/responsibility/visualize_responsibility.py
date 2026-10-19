"""A video of one agent's counterfactual responsibility through a clip: one
frame per context window k, stitched into a GIF and (with OpenCV) an MP4.

Every frame is drawn from the samples the pipeline itself scored -- the
values are the ones compute_responsibility.py computes under the same
settings (up to Monte-Carlo noise: the draws are fresh), not a re-estimate.

Each frame shows
  scene     the map, every agent at k with its last second of history, the
            queried agent's motion set (coloured by the model's mean per-token
            log-probability; samples the CVaR left out, e.g. by
            --route-consistent, in grey), its logged future, and the
            neighbours beta was taken over with their logged futures; the
            neighbour behind beta_s is outlined in red, the one behind beta_c
            in blue (purple when both)
  courtesy  that neighbour's samples behind beta_c:
              boltzmann  its pooled samples coloured by log P1/P2 (how much
                         the agent's logged future costs each one), and P1
                         against P2
              exact      its samples coloured by their term of the KL
                         (summed per-step KL), and those terms
  timeline  beta_s (m) and beta_c (nats) over the clip, the current window
            marked; with --hmm, the filtered level of every window (Eq. 6)
            as background colour

Two ways to run it:

  live      --scenario-index N [N ...]: computes the windows with the model
            (same flags as compute_responsibility.py, or --run DIR to take a
            run's settings) and saves them as record.pkl next to the video.
            The model is loaded once for every scenario and agent.
  offline   --record path/to/record.pkl [...]: renders saved records, no
            model or data needed.

One video goes to --out-dir; several go to --out-dir/<scenario>_<agent>.

The video shows the agent toward its own interacting neighbours (scope
"neighbours"); --direction is ignored.

Examples (from the repository root):
    python -m scripts.responsibility.visualize_responsibility \\
        --ckpt ckpts/clsft_E9.ckpt --data-root data/womd_cache/validation \\
        --scenario-index 0 --out-dir logs/responsibility/video_0
    python -m scripts.responsibility.visualize_responsibility \\
        --ckpt ckpts/clsft_E9.ckpt --data-root data/womd_cache/validation_interactive \\
        --scenario-index 3 7 --agent sdc adv --courtesy-estimator exact \\
        --neighbor-future hidden --hmm logs/responsibility/hmm.pkl --out-dir logs/responsibility/videos
    python -m scripts.responsibility.visualize_responsibility \\
        --record logs/responsibility/video_0/record.pkl --hmm logs/responsibility/hmm.pkl \\
        --out-dir logs/responsibility/video_0
"""

import argparse
import json
import pickle
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

from scripts.responsibility._cli import (  # noqa: E402
    add_model_args,
    add_responsibility_args,
    agent_type_from_args,
    responsibility_config_from_args,
)
from scripts.responsibility.visualize_motion_sets import box_corners, write_video  # noqa: E402

AGENT_COLOR = "#d62728"
NEIGHBOUR_COLOR = "#1f77b4"
OTHER_COLOR = "#9a9a9a"
SAFETY_COLOR = "#d62728"
COURTESY_COLOR = "#1f77b4"
BOTH_COLOR = "#7b2d8e"
# pt_token.pl_type: ["lane", "road_edge", "road_line", "crosswalk"] -> (colour, width)
MAP_STYLE = {0: ("#dcdcdc", 0.8), 1: ("#505050", 1.0), 2: ("#b0b0b0", 0.6), 3: ("#c9b458", 0.8)}
TYPE_NAME = {0: "veh", 1: "ped", 2: "cyc"}
RECORD_VERSION = 1
RUN_OVERRIDABLE = ("courtesy_estimator", "neighbor_future")  # flags that take precedence over --run


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scenario-index", type=int, nargs="+",
                     help="Live: scenario indices in --data-root (positions in its sorted file list).")
    src.add_argument("--scenario-file", nargs="+",
                     help="Live: scenario file names in --data-root, with or without .pkl (e.g. 12 for 12.pkl).")
    src.add_argument("--record", nargs="+", help="Offline: record.pkl files written by a live run.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hmm", default=None, help="hmm.pkl from fit_hmm.py: levels as the timeline's background.")
    p.add_argument("--view-radius", type=float, default=None, help="m; default: fitted to the clip (25-80 m).")
    p.add_argument("--fps", type=float, default=2.0)
    live = p.add_argument_group("live mode")
    live.add_argument("--data-root", default=None)
    live.add_argument("--agent", nargs="+", default=["sdc"],
                      help="sdc, adv (the first object of interest that is not the SDC) or an agent index; "
                           "several render each scenario for each.")
    live.add_argument("--run", default=None,
                      help="compute_responsibility.py --out-dir whose settings (config.json) to use "
                           "instead of the responsibility flags; --courtesy-estimator and "
                           "--neighbor-future, if given, still override it.")
    live.add_argument("--seed", type=int, default=0)
    add_model_args(p, ckpt_required=False)
    add_responsibility_args(p)
    args = p.parse_args()
    if args.record is None and (args.ckpt is None or args.data_root is None):
        p.error("live mode needs --ckpt and --data-root")
    if args.scenario_file is not None:
        from src.responsibility.results_io import scenario_files

        index = {f.stem: i for i, f in enumerate(scenario_files(args.data_root))}
        missing = [s for s in args.scenario_file if Path(s).stem not in index]
        if missing:
            p.error(f"not in {args.data_root}: {', '.join(missing)}")
        args.scenario_index = [index[Path(s).stem] for s in args.scenario_file]
    return args


# ----------------------------------------------------------------------------
# live capture
# ----------------------------------------------------------------------------


def config_from(args, num_historical_steps: int):
    """The responsibility settings and sampler of the run being shown."""
    from src.responsibility.interaction import InteractionConfig
    from src.responsibility.metrics import DIRECTION_OUTGOING, ResponsibilityConfig

    sampling = {"num_k": args.motion_set_num_k, "temp": args.motion_set_temp}
    if args.run is None:
        cfg = responsibility_config_from_args(args, num_historical_steps)
    else:
        saved = json.loads((Path(args.run) / "config.json").read_text())
        known = {f.name for f in fields(ResponsibilityConfig)}
        r = {k: v for k, v in saved["responsibility"].items() if k in known}
        r["interaction"] = InteractionConfig(**r.get("interaction", {}))
        cfg = ResponsibilityConfig(**r)
        sampling = saved.get("sampling", sampling)
        if cfg.num_historical_steps != num_historical_steps:
            raise SystemExit(f"{args.run} used num_historical_steps={cfg.num_historical_steps}, "
                             f"the model has {num_historical_steps}")
        # these two, when given explicitly, override the run's: the same
        # windows under another estimator or another neighbour-future mode
        given = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
        overrides = {f: getattr(args, f) for f in RUN_OVERRIDABLE if "--" + f.replace("_", "-") in given}
        if overrides:
            print(f"overriding {args.run}'s settings: {overrides}", flush=True)
            cfg = replace(cfg, **overrides)
    # the agent toward its own neighbours only; the reverse queries would
    # cost rollouts the video never shows
    return replace(cfg, query_direction=DIRECTION_OUTGOING), sampling


def pick_agent(data, which: str) -> int:
    role = data["agent"]["role"]
    if which == "sdc":
        idx = role[:, 0].nonzero().flatten()
    elif which == "adv":
        idx = (role[:, 1] & ~role[:, 0]).nonzero().flatten()
    else:
        return int(which)
    if idx.numel() == 0:
        raise ValueError(f"no {which} in this scenario")
    return int(idx[0])


def to_numpy(x):
    import torch

    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_numpy(v) for v in x)
    return x


def scene_snapshot(data) -> Dict:
    agent = data["agent"]
    return to_numpy({
        "position": agent["position"][..., :2],
        "heading": agent["heading"],
        "velocity": agent["velocity"],
        "valid": agent["valid_mask"],
        "shape": agent["shape"],
        "type": agent["type"],
        "id": agent["id"],
        "is_sdc": agent["role"][:, 0],
        "map_pos": data["map_save"]["traj_pos"],
        "map_type": data["pt_token"]["pl_type"],
    })


def live_records(args) -> Iterator[Tuple[str, Dict]]:
    """(name, record) for every scenario and agent asked for."""
    import torch

    from src.responsibility.dataset_builder import compute_scenario_responsibility, window_steps
    from src.responsibility.motion_set import MotionSetSynthesizer
    from src.responsibility.pipeline import (
        compose_responsibility_config,
        default_sampling_scheme,
        load_scenario,
        load_smart_model,
        resolve_device,
        scenario_id_of,
    )

    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    device = resolve_device(args.device)
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=device)
    cfg, sampling = config_from(args, hydra_cfg.model.model_config.decoder.num_historical_steps)
    synth = MotionSetSynthesizer(model, default_sampling_scheme(**sampling),
                                 rollout_batch_size=args.rollout_batch_size or None)
    several = len(args.scenario_index) * len(args.agent) > 1
    for index in args.scenario_index:
        data = load_scenario(args.data_root, index, device)
        scenario_id = scenario_id_of(data, str(index))
        for which in args.agent:
            name = f"{scenario_id}_{which}" if several else scenario_id
            try:
                a = pick_agent(data, which)
                if cfg.interaction.ooi_only and not bool(data["agent"]["role"][a, 1]):
                    raise ValueError(f"agent #{a} is not an object of interest (--ooi-neighbours)")
            except ValueError as e:
                print(f"{name}: skipped ({e})", flush=True)
                continue
            torch.manual_seed(args.seed)
            details: Dict = {}
            observations = compute_scenario_responsibility(
                synth, data, cfg, agent_type_filter=agent_type_from_args(args),
                steps=window_steps(data, cfg, synth, cfg.window_stride), agents=[a], details=details,
            )
            frames = []
            for obs in observations:
                if obs.agent_idx != a:
                    continue
                d = to_numpy(details[(obs.k, a)])
                frames.append({
                    "k": obs.k,
                    "beta": obs.beta.numpy(),
                    "neighbours": d["neighbours"],
                    "per_neighbour": {b: obs.per_neighbour[int(data["agent"]["id"][b])] for b in d["neighbours"]},
                    "interaction": {b: obs.interaction[int(data["agent"]["id"][b])] for b in d["neighbours"]},
                    "safety": d["safety"],
                    "courtesy": d["courtesy"],
                })
                s, c = frames[-1]["beta"]
                print(f"{name} k={obs.k:3d}: safety {s:+.3f} m, courtesy {c:.4f} nats, "
                      f"{len(d['neighbours'])} neighbour(s)", flush=True)
            yield name, {
                "version": RECORD_VERSION,
                "scenario_id": scenario_id,
                "scenario_index": index,
                "agent": a,
                "agent_id": int(data["agent"]["id"][a]),
                "config": asdict(cfg),
                "sampling": sampling,
                "ckpt": str(Path(args.ckpt).resolve()),
                "scene": scene_snapshot(data),
                "frames": frames,
            }


# ----------------------------------------------------------------------------
# rendering (from records only)
# ----------------------------------------------------------------------------


def map_segments(scene) -> Dict[Tuple[str, float], np.ndarray]:
    pos, kind = scene["map_pos"], scene["map_type"]
    out = {}
    for t, style in MAP_STYLE.items():
        pieces = pos[kind == t]
        if len(pieces):
            out[style] = np.concatenate([pieces[:, :-1, None], pieces[:, 1:, None]], axis=2).reshape(-1, 2, 2)
    return out


def draw_map(ax, groups, scale=1.0):
    for (colour, width), segs in groups.items():
        ax.add_collection(LineCollection(segs, colors=colour, linewidths=width * scale, zorder=1))


def draw_box(ax, scene, i, k, zorder=6, **style):
    """Agent i's footprint at k, its front edge drawn thick so the heading reads."""
    length, width = scene["shape"][i, :2]
    corners = box_corners(scene["position"][i, k], float(scene["heading"][i, k]), float(length), float(width))
    ax.add_patch(Polygon(corners, closed=True, zorder=zorder, **style))
    ax.plot(*corners[:2].T, color="black", lw=1.6, solid_capstyle="butt", zorder=zorder + 0.1)


def logged_future(scene, i, k, horizon):
    """Agent i's logged positions over [k, k + horizon] (from now, so the
    line starts at the agent)."""
    fut = slice(k, k + 1 + horizon)
    return scene["position"][i, fut][scene["valid"][i, fut]]


def from_now(scene, i, k, plans):  # [N, H, 2] -> [N, H + 1, 2], starting at agent i's pose at k
    start = np.broadcast_to(scene["position"][i, k], (plans.shape[0], 1, 2))
    return np.concatenate([start, plans], axis=1)


def worst(frame, slot: int) -> Tuple[Optional[int], float]:
    """The neighbour behind beta_s (slot 0) or beta_c (slot 1), or None when
    that term is zero."""
    items = [(b, v[slot]) for b, v in frame["per_neighbour"].items()]
    b, v = max(items, key=lambda bv: bv[1], default=(None, 0.0))
    return (b, v) if v > 0 else (None, v)


def horizon_of(frame, default: int) -> int:
    if frame["safety"] is not None:
        return frame["safety"]["samples"].shape[1]
    return default


def fit_radius(scene, agent, frames, horizon, lo=25.0, hi=80.0):
    need = 0.0
    for f in frames:
        centre = scene["position"][agent, f["k"]]
        pts = [scene["position"][f["neighbours"], f["k"]]]
        if f["safety"] is not None:
            pts.append(f["safety"]["samples"].reshape(-1, 2))
        pts.append(logged_future(scene, agent, f["k"], horizon))
        need = max(need, float(np.abs(np.concatenate(pts) - centre).max(initial=0.0)))
    return float(np.clip(need + 8.0, lo, hi))


def draw_scene(ax, scene, agent, frame, groups, radius, horizon):
    k = frame["k"]
    centre = scene["position"][agent, k]
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfbfb")
    draw_map(ax, groups)

    neighbours = set(frame["neighbours"])
    s_id, _ = worst(frame, 0)
    c_id, _ = worst(frame, 1)
    for i in range(scene["valid"].shape[0]):
        if not scene["valid"][i, k]:
            continue
        colour = AGENT_COLOR if i == agent else NEIGHBOUR_COLOR if i in neighbours else OTHER_COLOR
        hist = slice(max(0, k - 10), k + 1)
        h = scene["position"][i, hist][scene["valid"][i, hist]]
        if len(h) > 1:
            ax.plot(*h.T, color=colour, lw=0.8, alpha=0.5, zorder=3)
        edge, lw = "black", 0.5
        if i == s_id and i == c_id:
            edge, lw = BOTH_COLOR, 2.6
        elif i == s_id:
            edge, lw = SAFETY_COLOR, 2.2
        elif i == c_id:
            edge, lw = COURTESY_COLOR, 2.2
        draw_box(ax, scene, i, k, facecolor=colour, edgecolor=edge, lw=lw,
                 alpha=0.9 if i == agent or i in neighbours else 0.45)
        if i in neighbours:
            f = logged_future(scene, i, k, horizon)
            if len(f) > 1:
                ax.plot(*f.T, color=NEIGHBOUR_COLOR, lw=1.3, ls="--", zorder=5)
            ax.annotate(f"#{i}", scene["position"][i, k], fontsize=6.5, color=NEIGHBOUR_COLOR,
                        xytext=(4, 4), textcoords="offset points", zorder=7)
    if frame["per_neighbour"]:
        rows = [f"#{b} {TYPE_NAME.get(int(scene['type'][b]), '?')}: {v[0]:+.2f} m  {v[1]:.3f} nats"
                for b, v in sorted(frame["per_neighbour"].items(), key=lambda bv: -bv[1][0])]
        ax.text(0.99, 0.01, "per neighbour  beta_s  beta_c\n" + "\n".join(rows), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, family="monospace", zorder=9,
                bbox=dict(facecolor="white", alpha=0.85, lw=0))

    cmap = plt.get_cmap("plasma")
    safety = frame["safety"]
    if safety is not None:
        samples, lp = from_now(scene, agent, k, safety["samples"]), safety["log_prob"]
        kept = np.zeros(len(samples), dtype=bool)
        kept[safety["kept"]] = True
        norm = plt.Normalize(lp.min(), lp.max() if lp.max() > lp.min() else lp.min() + 1e-6)
        for j in np.flatnonzero(~kept):
            ax.plot(*samples[j].T, color="#b5b5b5", lw=1.0, alpha=0.7, zorder=5.5)
        for j in sorted(np.flatnonzero(kept), key=lambda j: lp[j]):
            ax.plot(*samples[j].T, color=cmap(norm(lp[j])), lw=1.6, alpha=0.8, zorder=5.5)
        ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, pad=0.01
                           ).set_label("mean per-token log p of sample", fontsize=7)
    logged = logged_future(scene, agent, k, horizon)
    if len(logged) > 1:
        ax.plot(*logged.T, color="black", lw=1.6, ls="--", zorder=8)
    n_kept = len(safety["kept"]) if safety is not None else 0
    n_all = len(safety["samples"]) if safety is not None else 0
    handles = [
        plt.Line2D([], [], color=cmap(0.8), lw=1.5, label=f"motion set ({n_kept}/{n_all} in the CVaR)"),
        plt.Line2D([], [], color="black", ls="--", lw=1.5, label="agent, logged"),
        plt.Line2D([], [], color=NEIGHBOUR_COLOR, ls="--", lw=1.3, label="neighbours, logged"),
        plt.Line2D([], [], color=SAFETY_COLOR, lw=2.2, label="beta_s from"),
        plt.Line2D([], [], color=COURTESY_COLOR, lw=2.2, label="beta_c toward"),
    ]
    if n_kept < n_all:
        handles.insert(1, plt.Line2D([], [], color="#b5b5b5", lw=1.0, label="left out of the CVaR"))
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, framealpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])


def frame_pair(scene, agent, b, k, plans, horizon, radius):
    """Centre and half-width of a courtesy panel: b's plans, the agent's
    logged future and both agents now."""
    shown = np.concatenate([plans.reshape(-1, 2), logged_future(scene, agent, k, horizon),
                            scene["position"][[b, agent], k]])
    lo, hi = shown.min(0), shown.max(0)
    return (lo + hi) / 2, float(np.clip((hi - lo).max() / 2 + 6.0, 15.0, min(radius, 60.0)))


def draw_pair(ax, scene, agent, b, k, horizon, groups, centre, r):
    draw_map(ax, groups, 0.8)
    a_fut = logged_future(scene, agent, k, horizon)
    if len(a_fut) > 1:
        ax.plot(*a_fut.T, color=AGENT_COLOR, lw=2.0, ls="--", zorder=5)
    b_fut = logged_future(scene, b, k, horizon)
    if len(b_fut) > 1:
        ax.plot(*b_fut.T, color=NEIGHBOUR_COLOR, lw=1.2, ls="--", zorder=5)
    draw_box(ax, scene, agent, k, facecolor=AGENT_COLOR, edgecolor="black", lw=0.8)
    draw_box(ax, scene, b, k, facecolor=NEIGHBOUR_COLOR, edgecolor="black", lw=0.5)
    ax.set_xlim(centre[0] - r, centre[0] + r)
    ax.set_ylim(centre[1] - r, centre[1] + r)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfbfb")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_courtesy(axes, scene, agent, frame, groups, radius, horizon, cfg):
    k = frame["k"]
    c_id, value = worst(frame, 1)
    if c_id is None or c_id not in frame["courtesy"]:
        if not frame["neighbours"]:
            text = "no interacting neighbour\n(no responsibility)"
        elif not frame["courtesy"]:
            skip = cfg.get("courtesy_skip_distance")
            text = ("every neighbour stays more than\n"
                    f"{skip:g} m away over the horizon:\ncourtesy not rolled out" if skip else
                    "no courtesy target")
        else:
            text = "courtesy is zero toward\nevery neighbour"
        axes[0].text(0.5, 0.5, text, ha="center", va="center", transform=axes[0].transAxes, fontsize=8)
        axes[0].axis("off")
        axes[1].axis("off")
        return
    pair = frame["courtesy"][c_id]
    H = pair["plans"].shape[1]
    plans = from_now(scene, c_id, k, pair["plans"])
    centre, r = frame_pair(scene, agent, c_id, k, plans, H, radius)
    ax, bar = axes
    draw_pair(ax, scene, agent, c_id, k, H, groups, centre, r)
    if pair["kind"] == "boltzmann":
        lp1, lp2 = pair["log_p_with"].astype(float), pair["log_p_without"].astype(float)
        shift = lp1 - lp2
        cmap, norm = plt.get_cmap("RdBu"), plt.Normalize(-3.0, 3.0)
        for j in np.argsort(-np.abs(shift)):
            ax.plot(*plans[j].T, color=cmap(norm(shift[j])), lw=1.3, alpha=0.85, zorder=3)
        ax.set_title(f"courtesy toward #{c_id} (Boltzmann, Eq. 7):\nits {len(plans)} pooled samples", fontsize=8)
        label = "log P1/P2 (the agent's future costs it: red)"
        p1, p2 = np.exp(lp1), np.exp(lp2)
        order = np.argsort(-p2)
        x = np.arange(len(order))
        bar.bar(x - 0.2, p2[order], width=0.4, color="#9a9a9a", label="P2: without the agent")
        bar.bar(x + 0.2, p1[order], width=0.4, color=COURTESY_COLOR, label="P1: with the agent")
        bar.set_xlabel(f"#{c_id}'s samples, by P2", fontsize=7)
        bar.set_ylabel("Boltzmann probability", fontsize=7)
        bar.legend(fontsize=6.5, loc="upper right")
        bar.set_title(f"KL(P1 || P2) = {value:.3f} nats", fontsize=8)
    else:
        per = pair["per_sample"].astype(float)
        cmap, norm = plt.get_cmap("viridis"), plt.Normalize(0.0, max(per.max(), 1e-6))
        for j in np.argsort(per):
            ax.plot(*plans[j].T, color=cmap(norm(per[j])), lw=1.3, alpha=0.85, zorder=3)
        ax.set_title(f"courtesy toward #{c_id} (exact KL, Eq. 4):\nits {len(plans)} samples with the agent",
                     fontsize=8)
        label = "sample's summed per-step KL (nats)"
        order = np.argsort(-per)
        bar.bar(np.arange(len(per)), per[order], color=[cmap(norm(v)) for v in per[order]])
        bar.axhline(value, color="black", lw=1.0, ls="--", label=f"mean = beta_c = {value:.3f}")
        bar.set_xlabel(f"#{c_id}'s samples, by their term", fontsize=7)
        bar.set_ylabel("KL along the sample (nats)", fontsize=7)
        bar.legend(fontsize=6.5, loc="upper right")
        bar.set_title(f"KL(p(xi_b | x) || p(xi_b | x^-a)) = {value:.3f} nats", fontsize=8)
    bar.set_xticks([])
    bar.tick_params(labelsize=6.5)
    bar.grid(axis="y", alpha=0.3)
    cax = ax.inset_axes([0.05, 0.05, 0.4, 0.025])
    cb = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=6)
    cax.tick_params(labelsize=6)


def levels_of(frames, hmm_path) -> Dict[int, int]:
    """k -> filtered level (Eq. 6) of the clip's own beta sequence."""
    if hmm_path is None or not frames:
        return {}
    with open(hmm_path, "rb") as f:
        hmm = pickle.load(f)
    seq = np.stack([f["beta"] for f in frames])
    return {f["k"]: int(z) for f, z in zip(frames, hmm.predict_level(seq))}


def draw_timeline(ax, frames, n, levels):
    t = np.array([f["k"] / 10 for f in frames])
    s = np.array([f["beta"][0] for f in frames])
    c = np.array([f["beta"][1] for f in frames])
    title = "responsibility over the clip"
    if levels:
        n_levels = max(levels.values()) + 1
        cmap = plt.get_cmap("viridis", max(n_levels, 2))
        half = (t[1] - t[0]) / 2 if len(t) > 1 else 0.25
        for f in frames:
            if f["k"] in levels:
                ax.axvspan(f["k"] / 10 - half, f["k"] / 10 + half, color=cmap(levels[f["k"]]), alpha=0.2, lw=0)
        seen = sorted(set(levels.values()))
        ax.legend(handles=[Patch(facecolor=cmap(z), alpha=0.4, label=f"level {z}") for z in seen],
                  fontsize=6.5, loc="upper right", ncol=len(seen))
        title += " (background: HMM level, Eq. 6)"
    ax.axhline(0, color="#999999", lw=0.7)
    ax.plot(t, s, color=SAFETY_COLOR, marker="o", ms=3, lw=1.3)
    ax.plot(t[n], s[n], "o", color="black", ms=6)
    ax.set_ylabel(r"$\beta_s$ [m]", color=SAFETY_COLOR)
    ax.set_xlabel("context time t_k [s]")
    ax.set_xlim(t.min() - 0.5, t.max() + 0.5)
    ax.set_ylim(min(0.0, s.min() * 1.15), max(0.5, s.max() * 1.15))
    tx = ax.twinx()
    tx.plot(t, c, color=COURTESY_COLOR, marker="s", ms=3, lw=1.3)
    tx.plot(t[n], c[n], "s", color="black", ms=6)
    tx.set_ylabel(r"$\beta_c$ [nats]", color=COURTESY_COLOR)
    tx.set_ylim(0, max(0.05, c.max() * 1.15))
    ax.set_title(title, fontsize=8.5)
    ax.grid(alpha=0.3)


def render(record, n, groups, radius, levels):
    scene, agent, frames, cfg = record["scene"], record["agent"], record["frames"], record["config"]
    frame = frames[n]
    k = frame["k"]
    horizon = horizon_of(frame, cfg["metric_horizon"])
    fig = plt.figure(figsize=(12, 8.2), dpi=100)
    grid = fig.add_gridspec(2, 3, width_ratios=[2.2, 1.2, 1.2], height_ratios=[3, 1.2], hspace=0.28, wspace=0.3)
    ax_scene = fig.add_subplot(grid[0, 0])
    draw_scene(ax_scene, scene, agent, frame, groups, radius, horizon)
    draw_courtesy((fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])), scene, agent, frame, groups,
                  radius, horizon, cfg)
    draw_timeline(fig.add_subplot(grid[1, :]), frames, n, levels)
    s_id, _ = worst(frame, 0)
    c_id, _ = worst(frame, 1)
    speed = float(np.linalg.norm(scene["velocity"][agent, k]))
    level = levels.get(k)
    fig.suptitle(
        f"scenario {record['scenario_id']}  agent #{agent} (id {record['agent_id']})  "
        f"t_k = {k / 10:.1f} s  speed {speed:.1f} m/s  |  neighbor_future = {cfg['neighbor_future']}, "
        f"courtesy = {cfg['courtesy_estimator']}, {horizon / 10:.1f} s horizon\n"
        f"safety {frame['beta'][0]:+.3f} m" + (f" (vs #{s_id})" if s_id is not None else "")
        + f"    courtesy {frame['beta'][1]:.4f} nats" + (f" (toward #{c_id})" if c_id is not None else "")
        + (f"    level {level}" if level is not None else ""),
        fontsize=9.5)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def render_record(record, out_dir: Path, hmm_path=None, view_radius=None, fps=2.0):
    """Renders every frame of a record to ``out_dir``/frames and stitches
    them into responsibility.gif (and .mp4 with OpenCV)."""
    frames = record["frames"]
    if not frames:
        raise ValueError("the record has no frames (the agent was never queried)")
    scene, agent = record["scene"], record["agent"]
    radius = view_radius or fit_radius(scene, agent, frames, record["config"]["metric_horizon"])
    groups = map_segments(scene)
    levels = levels_of(frames, hmm_path)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    images = []
    for n, frame in enumerate(frames):
        image = render(record, n, groups, radius, levels)
        plt.imsave(out_dir / "frames" / f"k_{frame['k']:03d}.png", image)
        images.append(image)
    gif, mp4 = write_video(images, out_dir, fps, stem="responsibility")
    return gif, mp4, radius


def load_record(path) -> Dict:
    with open(path, "rb") as f:
        record = pickle.load(f)
    if record.get("version") != RECORD_VERSION:
        raise SystemExit(f"{path}: record version {record.get('version')}, this script reads {RECORD_VERSION}")
    return record


def main():
    args = parse_args()
    root = Path(args.out_dir)
    if args.record:
        jobs: Iterator[Tuple[str, Dict]] = ((Path(r).parent.name or Path(r).stem, load_record(r)) for r in args.record)
        several = len(args.record) > 1
    else:
        from src.responsibility.results_io import atomic_pickle

        jobs = live_records(args)
        several = len(args.scenario_index) * len(args.agent) > 1
    for name, record in jobs:
        out = root / name if several else root
        out.mkdir(parents=True, exist_ok=True)
        if not args.record:
            atomic_pickle(record, out / "record.pkl")
        if not record["frames"]:
            print(f"{name}: no window queried agent #{record['agent']}; nothing to render", flush=True)
            continue
        gif, mp4, radius = render_record(record, out, args.hmm, args.view_radius, args.fps)
        print(f"{name}: {len(record['frames'])} frames in {out / 'frames'} (view half-width {radius:.0f} m), "
              f"{gif}" + (f", {mp4[0]}" if mp4 else " (no OpenCV: GIF only)"), flush=True)


if __name__ == "__main__":
    main()

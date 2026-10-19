"""Renders the counterfactual motion set (paper Eq. 1a) of one agent at every
context step k of a clip, and stitches the frames into a video.

Each frame re-anchors the clip at k (shifted_context), draws N samples of
the queried agent's future from the model -- coloured by their model
log-probability -- against its logged future, the logged future of the
neighbour it interacts with most, and the map. A strip below the scene traces
the safety responsibility towards that neighbour (Eq. 3) over the clip,
computed from the same samples over the paper's 2 s horizon.

By default the queried agent is the self-driving car and the neighbour is
the agent with the tightest logged interaction with it over the clip
(smallest post-encroachment time, then smallest footprint gap; see
interaction.geometric_interaction). The view follows the pair at one zoom
level for the whole video, wide enough for both and the drawn samples.

Example:
    python -m scripts.responsibility.visualize_motion_sets \
        --ckpt ckpts/clsft_E9.ckpt --data-root data/womd_cache/validation \
        --scenario-index 0 --out-dir logs/responsibility/video
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from src.responsibility.dataset_builder import window_steps  # noqa: E402
from src.responsibility.interaction import most_interactive_neighbour  # noqa: E402
from src.responsibility.metrics import ResponsibilityConfig, safety_responsibility_pair  # noqa: E402
from src.responsibility.motion_set import (  # noqa: E402
    NEIGHBOR_FUTURE_MODES,
    MotionSetSynthesizer,
    shifted_context,
)
from src.responsibility.pipeline import (  # noqa: E402
    compose_responsibility_config,
    default_sampling_scheme,
    load_scenario,
    load_smart_model,
    resolve_device,
    scenario_id_of,
)
from src.responsibility.reward_model import trajectory_log_prob  # noqa: E402

QUERY_COLOR = "#d62728"
NEIGHBOR_COLOR = "#1f77b4"
OTHER_COLOR = "#7f7f7f"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-config", default="smart")
    p.add_argument("--scenario-index", type=int, default=0)
    p.add_argument("--agent-index", type=int, default=None,
                   help="Queried agent; default: the self-driving car (else the closest vehicle pair).")
    p.add_argument("--neighbor-index", type=int, default=None,
                   help="Default: the agent interacting most tightly with the queried one.")
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--horizon", type=int, default=30, help="Rollout length @10Hz, multiple of 5.")
    p.add_argument("--stride", type=int, default=5, help="Context steps between frames.")
    p.add_argument("--neighbor-future", default="logged", choices=list(NEIGHBOR_FUTURE_MODES))
    p.add_argument("--view-radius", type=float, default=None,
                   help="Half-width of the view in metres; default: fit the pair and samples (25-80 m).")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--motion-set-num-k", type=int, default=2048)
    p.add_argument("--motion-set-temp", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="cuda, cpu, or auto (cuda if available).")
    return p.parse_args()


def logged_throughout(data, first_step: int, last_step: int):
    return data["agent"]["valid_mask"][:, first_step : last_step + 1].all(-1)


def closest_vehicle_pair(data, first_step: int, last_step: int):
    """The vehicle pair, both logged throughout [first_step, last_step],
    that comes closest to each other in that interval."""
    pos = data["agent"]["position"][:, first_step : last_step + 1, :2]
    valid = logged_throughout(data, first_step, last_step)
    candidates = torch.nonzero(valid & (data["agent"]["type"] == 0)).flatten().tolist()
    best = None
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            gap = torch.linalg.norm(pos[a] - pos[b], dim=-1).min().item()
            if gap > 1.0 and (best is None or gap < best[2]):
                best = (a, b, gap)
    if best is None:
        raise RuntimeError("no pair of vehicles is logged over the whole clip; pass --agent-index")
    return best[:2]


def default_query(data, first_step: int, last_step: int):
    sdc = torch.nonzero(data["agent"]["role"][:, 0] & logged_throughout(data, first_step, last_step)).flatten()
    return int(sdc[0]) if sdc.numel() else None


def box_corners(center, heading, length, width):
    c, s = np.cos(heading), np.sin(heading)
    dx = np.array([length, length, -length, -length]) / 2
    dy = np.array([width, -width, -width, width]) / 2
    return np.stack([center[0] + c * dx - s * dy, center[1] + s * dx + c * dy], axis=-1)


def view_center(frame):
    pts = [frame["query_now"][None], frame["samples"].reshape(-1, 2)]
    if frame["neighbor_now"] is not None:
        pts.append(frame["neighbor_now"][None])
    pts = np.concatenate(pts)
    return (pts.min(0) + pts.max(0)) / 2, pts


def fit_view_radius(frames, margin=8.0, lo=25.0, hi=80.0):
    """One half-width for the whole video: covers the pair and the drawn
    samples in every frame, so the zoom does not jump between frames."""
    need = 0.0
    for f in frames:
        center, pts = view_center(f)
        need = max(need, float(np.abs(pts - center).max()))
    return float(np.clip(need + margin, lo, hi))


def render_frame(frame, map_segments, view_radius, title, timeline):
    fig = plt.figure(figsize=(8, 10), dpi=100)
    grid = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.25)
    ax = fig.add_subplot(grid[0])
    cx, cy = view_center(frame)[0]
    ax.set_xlim(cx - view_radius, cx + view_radius)
    ax.set_ylim(cy - view_radius, cy + view_radius)
    ax.set_aspect("equal")
    ax.set_facecolor("#fafafa")

    ax.add_collection(LineCollection(map_segments, colors="#c8c8c8", linewidths=0.8, zorder=1))

    for agent in frame["agents"]:
        color = {"query": QUERY_COLOR, "neighbor": NEIGHBOR_COLOR}.get(agent["role"], OTHER_COLOR)
        if len(agent["history"]) > 1:
            ax.plot(*agent["history"].T, color=color, lw=1.0, alpha=0.6, zorder=3)
        ax.add_patch(Polygon(agent["box"], closed=True, facecolor=color,
                             edgecolor="black", lw=0.6, alpha=0.85 if agent["role"] != "other" else 0.45, zorder=5))

    # motion set: least likely drawn first so the likely samples sit on top
    order = np.argsort(frame["log_prob"])
    lp = frame["log_prob"]
    norm = plt.Normalize(vmin=lp.min(), vmax=lp.max() if lp.max() > lp.min() else lp.min() + 1e-6)
    cmap = plt.get_cmap("plasma")
    for i in order:
        ax.plot(*frame["samples"][i].T, color=cmap(norm(lp[i])), lw=1.8, alpha=0.8, zorder=4)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("mean per-step log p (model)")

    gt = frame["query_future"]
    if len(gt) > 1:
        ax.plot(*gt.T, color="black", lw=1.3, ls="--", zorder=6, label="queried agent, logged future")
    nb = frame["neighbor_future"]
    if len(nb) > 1:
        ax.plot(*nb.T, color=NEIGHBOR_COLOR, lw=1.3, ls="--", zorder=6, label="neighbour, logged future")
    ax.plot([], [], color=cmap(0.8), lw=1.5, label="counterfactual samples")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # responsibility over the clip so far; the full time range stays fixed
    tl = fig.add_subplot(grid[1])
    t_all, beta_all, upto = timeline["t"], timeline["beta"], timeline["upto"]
    tl.axhline(0.0, color="#999999", lw=0.8)
    tl.plot(t_all[: upto + 1], beta_all[: upto + 1], color=QUERY_COLOR, marker="o", ms=3, lw=1.5)
    tl.plot(t_all[upto], beta_all[upto], color="black", marker="o", ms=6)
    tl.set_xlim(t_all[0] - 0.5, t_all[-1] + 0.5)
    span = max(0.5, float(np.abs(beta_all).max()) * 1.15)
    tl.set_ylim(-0.1 * span, span)
    tl.set_xlabel("context time t = k / 10 [s]")
    tl.set_ylabel(r"$\beta_{safety}$ [m]")
    tl.set_title(timeline["label"], fontsize=9)
    tl.grid(alpha=0.3)

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def write_video(frames, out_dir: Path, fps: float, stem: str = "motion_sets"):
    from PIL import Image

    gif_path = out_dir / f"{stem}.gif"
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(gif_path, save_all=True, append_images=pil[1:], duration=int(1000 / fps), loop=0)

    mp4_path = None
    try:
        import cv2

        h, w = frames[0].shape[:2]
        for codec in ("avc1", "mp4v"):
            path = out_dir / f"{stem}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
            if writer.isOpened():
                for f in frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                mp4_path = (path, codec)
                break
    except ImportError:
        pass
    return gif_path, mp4_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    device = resolve_device(args.device)
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=device)
    num_hist = hydra_cfg.model.model_config.decoder.num_historical_steps

    synth = MotionSetSynthesizer(
        model,
        default_sampling_scheme(num_k=args.motion_set_num_k, temp=args.motion_set_temp),
        rollout_batch_size=args.n_samples,
    )
    horizon = synth.round_horizon(args.horizon)
    # windows are laid out for the rollout length we draw ...
    window_cfg = ResponsibilityConfig(num_historical_steps=num_hist, metric_horizon=horizon)
    # ... while the annotated responsibility uses the paper's 2 s horizon
    metric_cfg = ResponsibilityConfig(num_historical_steps=num_hist)

    data = load_scenario(args.data_root, args.scenario_index, device)
    scenario_id = scenario_id_of(data)

    steps = window_steps(data, window_cfg, synth, stride=args.stride)
    n_step_total = data["agent"]["position"].shape[1]
    first, last = steps[0], min(n_step_total - 1, steps[-1] + horizon)
    a = args.agent_index if args.agent_index is not None else default_query(data, first, last)
    j = args.neighbor_index
    if a is None:
        a, j = closest_vehicle_pair(data, first, last)
        print(f"no self-driving car logged over the clip; closest vehicle pair {a}, {j}")
    if j is None:
        picked = most_interactive_neighbour(data, a, first, last)
        if picked is None:
            raise RuntimeError(f"agent {a} has no neighbour logged over the clip; pass --neighbor-index")
        j, scores = picked
        print(f"queried agent {a}, neighbour {j}: PET {scores['pet']:.1f} s, "
              f"closest footprint gap {scores['min_gap']:.1f} m")
    else:
        print(f"queried agent {a}, neighbour {j}")

    map_pos = data["map_save"]["traj_pos"].cpu().numpy()  # [n_pl, 3, 2] polyline pieces
    map_segments = np.concatenate([map_pos[:, :-1, None], map_pos[:, 1:, None]], axis=2).reshape(-1, 2, 2)

    frames = []
    for k in steps:
        with shifted_context(data, k, num_hist) as shifted:
            agent = shifted["agent"]
            now = num_hist - 1
            if not bool(agent["valid_mask"][a, now]):
                print(f"k={k}: queried agent not observed, skipping")
                continue

            tok_map, tok_agent = synth.tokenize(shifted)
            out = synth.sample_rollouts(
                tok_map, tok_agent, free_agent_indices=[a], n_samples=args.n_samples,
                rollout_horizon=horizon, neighbor_future=args.neighbor_future,
            )
            samples = out["pred_traj_10hz"][:, 0]  # [N, H, 2]
            log_prob = trajectory_log_prob(
                out["next_token_logits"][:, 0], out["next_token_idx"][:, 0], out["next_token_valid"][:, 0]
            )

            pos, head, valid = agent["position"][..., :2], agent["heading"], agent["valid_mask"]
            shape = agent["shape"]
            fut = slice(num_hist, num_hist + horizon)
            mh = metric_cfg.metric_horizon
            beta = safety_responsibility_pair(
                samples[:, :mh], pos[a, fut][:mh], pos[j, fut][:mh], valid[j, fut][:mh],
                metric_cfg.cvar_alpha, d_sat=metric_cfg.d_sat,
            )

            agents = []
            for i in range(pos.shape[0]):
                if not bool(valid[i, now]):
                    continue
                role = "query" if i == a else "neighbor" if i == j else "other"
                hist_mask = valid[i, : now + 1]
                agents.append({
                    "role": role,
                    "box": box_corners(pos[i, now].cpu().numpy(), float(head[i, now]), float(shape[i, 0]), float(shape[i, 1])),
                    "history": pos[i, : now + 1][hist_mask].cpu().numpy(),
                })
            frames.append({
                "k": k,
                "beta": float(beta),
                "query_now": pos[a, now].cpu().numpy(),
                "neighbor_now": pos[j, now].cpu().numpy() if bool(valid[j, now]) else None,
                "agents": agents,
                "samples": samples.cpu().numpy(),
                "log_prob": log_prob.cpu().numpy(),
                "query_future": pos[a, fut][valid[a, fut]].cpu().numpy(),
                "neighbor_future": pos[j, fut][valid[j, fut]].cpu().numpy(),
            })
        print(f"k={k:3d}: sampled, beta_safety={float(beta):+.3f} m", flush=True)

    if not frames:
        raise RuntimeError("no frames rendered")
    radius = args.view_radius or fit_view_radius(frames)
    t_all = np.array([f["k"] / 10 for f in frames])
    beta_all = np.array([f["beta"] for f in frames])
    label = (f"safety responsibility of agent {a} towards agent {j} "
             f"(CVaR$_{{{metric_cfg.cvar_alpha}}}$ over 2 s, d_sat {metric_cfg.d_sat:g} m)")
    images = []
    for n, frame in enumerate(frames):
        k = frame["k"]
        title = (f"scenario {scenario_id}   k = {k}  (t = {k / 10:.1f} s)\n"
                 f"{args.n_samples} samples of agent {a}'s next {horizon / 10:.1f} s, "
                 f"neighbor_future = {args.neighbor_future}")
        timeline = {"t": t_all, "beta": beta_all, "upto": n, "label": label}
        image = render_frame(frame, map_segments, radius, title, timeline)
        plt.imsave(out_dir / "frames" / f"k_{k:03d}.png", image)
        images.append(image)

    gif_path, mp4 = write_video(images, out_dir, args.fps)
    print(f"wrote {len(images)} frames to {out_dir / 'frames'} (view half-width {radius:.0f} m)")
    print(f"gif: {gif_path}")
    print(f"mp4: {mp4[0]} (codec {mp4[1]})" if mp4 else "mp4: opencv unavailable, gif only")


if __name__ == "__main__":
    main()

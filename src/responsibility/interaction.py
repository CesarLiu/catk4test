"""Which agents actually interact with a queried agent.

Responsibility is a max over neighbours (Eq. 5), so the neighbour set
decides what it measures. A plain radius around the current position both
misses real conflicts (a fast vehicle 40 m away that crosses the queried
agent's path a second later) and admits non-interactions (a car 10 m ahead
in the same lane that never closes in, whose safety term saturates to zero
anyway but still costs rollouts). This module scores candidates with two
kinds of evidence:

Geometric (from the log, no model calls), over the same window [k, k+H]
that responsibility itself reads, so selection adds no information beyond
what beta already uses:

  min_gap  closest approach between the two footprints (circle cover)
  pet      post-encroachment time: the smallest time difference with which
           the two agents occupy the same place (within a tolerance) --
           catches crossing and merging conflicts that never come close
           at the same instant
  ttc      time to collision at k under constant velocity

Model-based influence of b on a:

  I(b -> a) = mean_t KL( p(u_a,t | x) || p(u_a,t | x^{-b}) )

the exact divergence, step by step along a's *logged* trajectory (every
agent teacher-forced), between a's next-token distribution with and without
b in the scene. It is the approach of the paper's reference [23]
(conditional behaviour prediction), and it is Eq. (4)'s courtesy
responsibility of b toward a evaluated along the log instead of over
samples. It is cheap: one "b removed" copy yields the distributions of every
queried agent at once, so a whole influence matrix costs 1 + n_candidates
scene copies in one packed model call.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from omegaconf import OmegaConf
from torch import Tensor

from src.responsibility.geometry import circle_centers, circle_decomposition
from src.responsibility.motion_set import NEIGHBOR_FUTURE_LOGGED, MotionSetSynthesizer, RolloutJob

SELECTION_RADIUS = "radius"
SELECTION_GEOMETRIC = "geometric"
SELECTION_INFLUENCE = "influence"
SELECTION_MODES = (SELECTION_RADIUS, SELECTION_GEOMETRIC, SELECTION_INFLUENCE)

QUERY_VEHICLES = "vehicles"
QUERY_SDC = "sdc"
QUERY_INTEREST = "interest"
QUERY_ALL = "all"
QUERY_MODES = (QUERY_VEHICLES, QUERY_SDC, QUERY_INTEREST, QUERY_ALL)

DT = 0.1  # seconds per 10Hz step


@dataclass
class InteractionConfig:
    mode: str = SELECTION_GEOMETRIC
    # candidates must be within this distance at the current step; generous,
    # since the real filtering happens on the interaction evidence below
    prefilter_radius: float = 50.0
    # a candidate interacts if ANY of the geometric criteria fires ...
    gap_threshold: float = 10.0  # m, footprint gap; matches the default d_sat
    # For a car-following pair PET *is* the time headway, so this doubles as
    # the two-second rule; for crossings, ~2 s is a common potential-conflict
    # threshold. At 3 s a follower cruising 30 m behind at 10 m/s would count.
    pet_threshold: float = 2.0  # s
    pet_path_tolerance: float = 2.0  # m: two path points are "the same place"
    ttc_threshold: float = 4.0  # s
    ttc_horizon: float = 6.0  # s of constant-velocity extrapolation for TTC
    # ... or, in influence mode, if its influence on the queried agent does
    influence_threshold: float = 0.02  # nats per step
    max_neighbors: Optional[int] = 8  # keep the strongest; None keeps all


def select_query_agents(
    data,
    at_step: int,
    mode: str = QUERY_VEHICLES,
    agent_type: Optional[int] = 0,
) -> List[int]:
    """Agents whose responsibility is evaluated, among those observed at
    ``at_step``: every vehicle (``agent_type``), only the self-driving car
    (WOMD ``role[:, 0]``), the objects of interest / tracks to predict
    (``role[:, 1:]``), or everyone."""
    if mode not in QUERY_MODES:
        raise ValueError(f"query mode must be one of {QUERY_MODES}, got {mode!r}")
    agent = data["agent"]
    observed = agent["valid_mask"][:, at_step]
    if mode == QUERY_SDC:
        mask = agent["role"][:, 0] & observed
    elif mode == QUERY_INTEREST:
        mask = agent["role"][:, 1:].any(-1) & observed
    else:
        mask = observed.clone()
        if mode == QUERY_VEHICLES and agent_type is not None:
            mask &= agent["type"] == agent_type
    return mask.nonzero(as_tuple=True)[0].tolist()


# ---------------------------------------------------------------------------
# geometric evidence
# ---------------------------------------------------------------------------


def _window(data, idx: Tensor, first: int, last: int):
    agent = data["agent"]
    return (
        agent["position"][idx, first : last + 1, :2],
        agent["heading"][idx, first : last + 1],
        agent["valid_mask"][idx, first : last + 1],
        agent["shape"][idx],
    )


def geometric_interaction(
    data,
    query: int,
    candidates: Sequence[int],
    num_historical_steps: int,
    horizon: int,
    cfg: InteractionConfig,
    n_circles: int = 3,
) -> Dict[str, Tensor]:
    """min_gap [M] (m) and pet [M] (s) over the logged window up to
    ``horizon`` steps ahead (PET also looks back over the observed history),
    and ttc [M] (s) from constant-velocity extrapolation of the current state;
    inf where a criterion never applies."""
    device = data["agent"]["position"].device
    now = num_historical_steps - 1
    last = min(now + horizon, data["agent"]["position"].shape[1] - 1)
    cand = torch.as_tensor(list(candidates), dtype=torch.long, device=device)
    q = torch.as_tensor([query], dtype=torch.long, device=device)
    m = cand.numel()
    inf = torch.full((m,), float("inf"), device=device)
    if m == 0:
        return {"min_gap": inf, "pet": inf, "ttc": inf}

    q_pos, q_head, q_valid, q_shape = _window(data, q, now, last)
    c_pos, c_head, c_valid, c_shape = _window(data, cand, now, last)

    q_off, q_r = circle_decomposition(q_shape[:, 0], q_shape[:, 1], n_circles)
    c_off, c_r = circle_decomposition(c_shape[:, 0], c_shape[:, 1], n_circles)

    def footprint_gap(qp, qh, cp, ch):  # [1|M, T, 2], [1|M, T] -> [M, T]
        q_c = circle_centers(qp, qh, q_off)  # [1, T, C, 2]
        c_c = circle_centers(cp, ch, c_off)  # [M, T, C, 2]
        centre = torch.linalg.norm(q_c[:, :, :, None, :] - c_c[:, :, None, :, :], dim=-1)  # [M, T, C, C]
        return centre.amin(dim=(-2, -1)) - (q_r[0] + c_r)[:, None]

    # closest approach of the two footprints over the logged window
    gap = footprint_gap(q_pos, q_head, c_pos, c_head)
    min_gap = gap.masked_fill(~(q_valid & c_valid), float("inf")).amin(dim=-1).clamp(min=0.0)

    # PET: smallest time difference with which the two occupy the same place.
    # Observed history counts too (the other car may have cleared the spot a
    # moment ago), but at least one of the two occupations must lie in the
    # future: a conflict that fully resolved before now is not this window's.
    first = max(0, now - (num_historical_steps - 1))
    hq_pos, _, hq_valid, _ = _window(data, q, first, last)
    hc_pos, _, hc_valid, _ = _window(data, cand, first, last)
    together = (torch.cdist(hq_pos.expand(m, -1, -1), hc_pos) < cfg.pet_path_tolerance)
    together &= hq_valid[:, :, None] & hc_valid[:, None, :]
    steps = torch.arange(hq_pos.shape[1], device=device)
    future = (steps[:, None] >= now - first) | (steps[None, :] >= now - first)  # [T, T]
    dt = (steps[:, None] - steps[None, :]).abs().to(hq_pos.dtype) * DT
    pet = dt.expand(m, -1, -1).masked_fill(~(together & future), float("inf")).amin(dim=(-2, -1))

    # TTC: first time the footprints (same circle cover) touch when both keep
    # their current velocity and heading. Bounding circles are no good here:
    # two cars side by side in adjacent lanes would already "collide".
    agent = data["agent"]
    t = torch.arange(int(round(cfg.ttc_horizon / DT)) + 1, device=device, dtype=hq_pos.dtype) * DT
    q_ext = agent["position"][q, now, :2][:, None, :] + agent["velocity"][q, now][:, None, :] * t[None, :, None]
    c_ext = agent["position"][cand, now, :2][:, None, :] + agent["velocity"][cand, now][:, None, :] * t[None, :, None]
    q_h = agent["heading"][q, now][:, None].expand(-1, t.numel())
    c_h = agent["heading"][cand, now][:, None].expand(-1, t.numel())
    touching = footprint_gap(q_ext, q_h, c_ext, c_h) <= 0  # [M, S]
    first_touch = torch.where(touching, t[None, :], torch.full_like(t[None, :], float("inf"))).amin(dim=-1)
    now_valid = agent["valid_mask"][cand, now] & agent["valid_mask"][query, now]
    ttc = first_touch.masked_fill(~now_valid, float("inf"))

    return {"min_gap": min_gap, "pet": pet, "ttc": ttc}


def geometric_criteria(scores: Dict[str, Tensor], cfg: InteractionConfig) -> Tensor:  # [M] bool
    return (
        (scores["min_gap"] <= cfg.gap_threshold)
        | (scores["pet"] <= cfg.pet_threshold)
        | (scores["ttc"] <= cfg.ttc_threshold)
    )


# ---------------------------------------------------------------------------
# model-based influence
# ---------------------------------------------------------------------------


def influence_matrix(
    synthesizer: MotionSetSynthesizer,
    tokenized_map: Dict[str, Tensor],
    tokenized_agent: Dict[str, Tensor],
    queried: Sequence[int],
    candidates: Sequence[int],
    rollout_horizon: Optional[int] = None,
) -> Tensor:  # [len(queried), len(candidates)]
    """I[i, j] = mean over the steps where queried[i] is logged of
    KL( p(u_t | x) || p(u_t | x^{-candidates[j]}) ) for queried[i]'s next
    token, with every agent replaying its log (so each step conditions only
    on what had happened by then). Entries with i == j are 0."""
    queried, candidates = list(queried), list(candidates)
    if not queried or not candidates:
        return torch.zeros(len(queried), len(candidates), device=tokenized_agent["valid_mask"].device)

    outputs = tuple(queried)
    jobs = [RolloutJob(free=(), n_samples=1, outputs=outputs)]
    jobs += [RolloutJob(free=(), n_samples=1, removed=(b,), outputs=outputs) for b in candidates]
    # argmax only matters where a log runs out and a queried agent's token is
    # not forced; keeping it deterministic keeps the two contexts comparable
    argmax = OmegaConf.create({"criterium": "topk_prob", "num_k": 1, "temp": 1.0})
    results = synthesizer.run_jobs(
        tokenized_map,
        tokenized_agent,
        jobs,
        sampling_scheme=argmax,
        rollout_horizon=rollout_horizon,
        neighbor_future=NEIGHBOR_FUTURE_LOGGED,
        keep_logits=True,
    )

    base = torch.log_softmax(results[0]["next_token_logits"][0], dim=-1)  # [Q, S, V]
    n_steps = base.shape[1]
    q_idx = torch.as_tensor(queried, device=base.device)
    logged = tokenized_agent["valid_mask"][q_idx, 2 : 2 + n_steps].to(base.dtype)  # [Q, S]
    denom = logged.sum(-1).clamp(min=1.0)

    columns = []
    for j, b in enumerate(candidates):
        other = torch.log_softmax(results[j + 1]["next_token_logits"][0], dim=-1)
        kl_t = (base.exp() * (base - other)).sum(-1).clamp(min=0.0)  # [Q, S]
        col = (kl_t * logged).sum(-1) / denom
        col = torch.where(q_idx == b, torch.zeros_like(col), col)
        columns.append(col)
    return torch.stack(columns, dim=1)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


@dataclass
class NeighbourSelection:
    neighbours: Dict[int, List[int]]
    # per queried agent: candidate -> {"min_gap", "pet", "ttc", "influence", "selected"}
    diagnostics: Dict[int, Dict[int, Dict[str, float]]]


def select_interacting_neighbours(
    synthesizer: Optional[MotionSetSynthesizer],
    data,
    queried: Sequence[int],
    num_historical_steps: int,
    horizon: int,
    cfg: InteractionConfig,
    tokenized_map: Optional[Dict[str, Tensor]] = None,
    tokenized_agent: Optional[Dict[str, Tensor]] = None,
    radius: Optional[float] = None,
) -> NeighbourSelection:
    """Neighbour sets for every queried agent at the current step.

    - ``radius``: everyone observed within ``radius`` now (the old rule)
    - ``geometric``: prefiltered candidates for which any of min_gap, PET or
      TTC falls under its threshold, strongest first
    - ``influence``: the geometric set plus any candidate whose model
      influence exceeds ``influence_threshold``, ranked by influence
    """
    if cfg.mode not in SELECTION_MODES:
        raise ValueError(f"selection mode must be one of {SELECTION_MODES}, got {cfg.mode!r}")
    agent = data["agent"]
    now = num_historical_steps - 1
    observed = agent["valid_mask"][:, now]
    pos_now = agent["position"][:, now, :2]

    def prefilter(a: int, r: float) -> List[int]:
        dist = torch.linalg.norm(pos_now - pos_now[a], dim=-1)
        mask = (dist <= r) & observed & observed[a]
        mask[a] = False
        return mask.nonzero(as_tuple=True)[0].tolist()

    if cfg.mode == SELECTION_RADIUS:
        r = radius if radius is not None else cfg.prefilter_radius
        nbrs = {a: prefilter(a, r) for a in queried}
        return NeighbourSelection(nbrs, {a: {b: {"selected": 1.0} for b in n} for a, n in nbrs.items()})

    candidates = {a: prefilter(a, cfg.prefilter_radius) for a in queried}
    influence = {}
    if cfg.mode == SELECTION_INFLUENCE:
        if synthesizer is None or tokenized_map is None or tokenized_agent is None:
            raise ValueError("influence selection needs the synthesizer and the tokenized scene")
        union = sorted({b for c in candidates.values() for b in c})
        matrix = influence_matrix(
            synthesizer, tokenized_map, tokenized_agent, list(queried), union,
            rollout_horizon=synthesizer.round_horizon(horizon),
        )
        col = {b: j for j, b in enumerate(union)}
        influence = {a: {b: float(matrix[i, col[b]]) for b in candidates[a]} for i, a in enumerate(queried)}

    neighbours, diagnostics = {}, {}
    for a in queried:
        cands = candidates[a]
        scores = geometric_interaction(data, a, cands, num_historical_steps, horizon, cfg)
        geo_ok = geometric_criteria(scores, cfg)
        diag, keep = {}, []
        for j, b in enumerate(cands):
            infl = influence.get(a, {}).get(b, float("nan"))
            ok = bool(geo_ok[j]) or (cfg.mode == SELECTION_INFLUENCE and infl >= cfg.influence_threshold)
            diag[b] = {
                "min_gap": float(scores["min_gap"][j]),
                "pet": float(scores["pet"][j]),
                "ttc": float(scores["ttc"][j]),
                "influence": infl,
                "selected": 0.0,
            }
            if ok:
                keep.append(b)
        if cfg.mode == SELECTION_INFLUENCE:
            keep.sort(key=lambda b: -diag[b]["influence"])
        else:  # closest approach first
            keep.sort(key=lambda b: diag[b]["min_gap"])
        if cfg.max_neighbors is not None:
            keep = keep[: cfg.max_neighbors]
        for b in keep:
            diag[b]["selected"] = 1.0
        neighbours[a], diagnostics[a] = keep, diag
    return NeighbourSelection(neighbours, diagnostics)


def most_interactive_neighbour(
    data,
    query: int,
    first_step: int,
    last_step: int,
    cfg: Optional[InteractionConfig] = None,
    min_presence: float = 0.5,
) -> Optional[Tuple[int, Dict[str, float]]]:
    """The agent whose logged motion over [first_step, last_step] interacts
    most tightly with ``query``, for picking one pair to show: smallest
    post-encroachment time (someone passed through the same spot shortly
    before or after), then closest footprint gap. Only agents logged for at
    least ``min_presence`` of the interval are considered. Returns
    ``(index, {"min_gap", "pet", "ttc"})`` or None."""
    valid = data["agent"]["valid_mask"][:, first_step : last_step + 1]
    present = torch.nonzero(valid.float().mean(-1) >= min_presence).flatten().tolist()
    candidates = [i for i in present if i != query]
    if not candidates:
        return None
    scores = geometric_interaction(
        data, query, candidates, num_historical_steps=first_step + 1,
        horizon=last_step - first_step, cfg=cfg or InteractionConfig(),
    )
    best = min(range(len(candidates)), key=lambda m: (float(scores["pet"][m]), float(scores["min_gap"][m])))
    return candidates[best], {k: float(v[best]) for k, v in scores.items()}

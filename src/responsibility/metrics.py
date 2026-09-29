"""Safety and courtesy responsibility (paper Sec. III-B, Eq. 2-5, and the
tractable Boltzmann approximation to courtesy responsibility from Sec. V-B,
Eq. 7, which we use in place of the intractable exact KL of Eq. 4).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from src.responsibility.geometry import (
    BoxTrajectories,
    collision_cost,
    pairwise_min_distance_over_time,
    road_deviation_cost,
)
from src.responsibility.interaction import InteractionConfig
from src.responsibility.motion_set import (
    NEIGHBOR_FUTURE_LOGGED,
    NEIGHBOR_FUTURE_SIMULATED,
    MotionSetSynthesizer,
    RolloutJob,
    take_samples,
)
from src.responsibility.risk import cvar


COURTESY_BOLTZMANN = "boltzmann"
COURTESY_EXACT = "exact"
COURTESY_ESTIMATORS = (COURTESY_BOLTZMANN, COURTESY_EXACT)

DIRECTION_OUTGOING = "outgoing"
DIRECTION_INCOMING = "incoming"
DIRECTION_BOTH = "both"
DIRECTIONS = (DIRECTION_OUTGOING, DIRECTION_INCOMING, DIRECTION_BOTH)


@dataclass
class ResponsibilityConfig:
    n_safety_samples: int = 40
    cvar_alpha: float = 0.1
    n_courtesy_samples: int = 20
    # 30 m rather than SMART's own 60 m a2a_radius: beyond roughly this range
    # the "alternatives" of a queried agent gain distance against a neighbour
    # trivially, which swamps the metric with non-interactions.
    neighbor_radius: float = 30.0
    # Interaction range at which D_g saturates (see geometry.saturate). None
    # reproduces the paper's literal, unsaturated definition.
    d_sat: Optional[float] = 10.0
    # Steps of future over which responsibility is measured. Fixing this
    # keeps beta comparable across context times k (see dataset_builder's
    # sliding windows), where the amount of remaining logged future varies.
    # 20 steps @10Hz = 2 s, matching the paper's prediction horizon.
    metric_horizon: int = 20
    # Spacing between the context steps at which responsibility is evaluated
    # within one clip; 5 steps = 0.5 s. Smaller gives longer beta sequences
    # for the HMM to learn transitions from, at proportional cost.
    window_stride: int = 5
    # Must match the loaded checkpoint's decoder.num_historical_steps; it
    # defines where the logged future starts.
    num_historical_steps: int = 11
    # How much of the neighbours' future the queried agent may condition on
    # while its alternatives are sampled; see motion_set.NEIGHBOR_FUTURE_MODES.
    # "logged" is faithful to "b replays its log" but lets the queried agent
    # foresee the real scene, which biases safety responsibility upward;
    # "hidden" is the paper's stated open-loop reading.
    neighbor_future: str = NEIGHBOR_FUTURE_LOGGED
    n_collision_circles: int = 3
    collision_margin: float = 0.0
    collision_weight: float = 10.0  # matches paper's R = -10*c_coll - c_dev
    drivable_half_width: float = 3.0
    # Whose responsibility is measured: "vehicles" (every vehicle), "sdc" (the
    # self-driving car only), "interest" (WOMD objects of interest / tracks
    # to predict) or "all"; see interaction.select_query_agents.
    query_mode: str = "vehicles"
    # With query_mode="sdc": "outgoing" measures the SDC toward the agents it
    # interacts with, "incoming" measures each of those agents toward the
    # SDC, "both" does both (kept apart by ResponsibilityObservation.scope).
    query_direction: str = DIRECTION_OUTGOING
    # How the neighbour set of Eq. (5) is chosen; see interaction.py.
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    # How courtesy (Eq. 4) is estimated: "boltzmann" is the paper's Sec. V-B
    # proxy (Eq. 7); "exact" is the KL between the model's own distributions
    # over the target's motion with and without the queried agent, which a
    # tokenized model makes computable (see exact_courtesy_from_logits).
    courtesy_estimator: str = COURTESY_BOLTZMANN
    # Pairs further apart than this are skipped before rolling out. For the
    # Boltzmann proxy this is exact (its two rewards then coincide); for the
    # exact estimator it is an approximation that drops weak far-field
    # influence. None disables it.
    courtesy_skip_distance: Optional[float] = 20.0


def future_slice(tensor: Tensor, num_historical_steps: int, horizon: Optional[int] = None) -> Tensor:
    """Slices the logged future: the ``horizon`` steps that follow the
    current step (index ``num_historical_steps - 1``). Slicing forward from
    an explicit origin rather than backward from the end keeps this correct
    under the shifted contexts used for sliding-window responsibility."""
    end = None if horizon is None else num_historical_steps + horizon
    return tensor[:, num_historical_steps:end]


def effective_horizon(data, cfg: "ResponsibilityConfig", rollout_len: Optional[int] = None) -> int:
    """Steps of future actually usable: the configured metric horizon,
    capped by how much logged future the clip still has after the current
    step and by how far the model rolled out."""
    available = data["agent"]["position"].shape[1] - cfg.num_historical_steps
    horizon = min(cfg.metric_horizon, available)
    if rollout_len is not None:
        horizon = min(horizon, rollout_len)
    return max(horizon, 0)


def ground_truth_boxes(
    data, indices: Sequence[int], num_historical_steps: int, horizon: Optional[int] = None
) -> BoxTrajectories:
    """Logged future boxes of the given agents over the next ``horizon`` steps."""
    device = data["agent"]["position"].device
    idx = torch.as_tensor(list(indices), dtype=torch.long, device=device)
    shape = data["agent"]["shape"][idx]
    return BoxTrajectories(
        pos=future_slice(data["agent"]["position"][idx, :, :2], num_historical_steps, horizon),
        heading=future_slice(data["agent"]["heading"][idx], num_historical_steps, horizon),
        valid=future_slice(data["agent"]["valid_mask"][idx], num_historical_steps, horizon),
        length=shape[:, 0],
        width=shape[:, 1],
    )


def rollout_boxes(
    rollouts: Dict[str, Tensor],
    slot: int,
    shape: Tensor,  # [3] of the free agent
    horizon: Optional[int] = None,
) -> BoxTrajectories:
    """Sampled rollouts of one free agent as boxes. A rollout is valid over
    its whole horizon by construction; neighbour validity is handled by the
    other side of the collision computation."""
    pos = rollouts["pred_traj_10hz"][:, slot]  # [N, T, 2]
    head = rollouts["pred_head_10hz"][:, slot]  # [N, T]
    n, t = pos.shape[0], pos.shape[1]
    boxes = BoxTrajectories(
        pos=pos,
        heading=head,
        valid=torch.ones(n, t, dtype=torch.bool, device=pos.device),
        length=shape[0].expand(n),
        width=shape[1].expand(n),
    )
    return boxes.truncate(horizon) if horizon is not None else boxes


def safety_responsibility_pair(
    sampled_traj_a: Tensor,  # [N, T, 2]
    gt_traj_a: Tensor,  # [T, 2]
    gt_traj_b: Tensor,  # [T, 2]
    valid_b: Tensor,  # [T]
    alpha: float,
    d_sat: Optional[float] = None,
) -> Tensor:  # scalar
    """beta^safety_k(a, b, x_k), Eq. (3): CVaR_alpha of the safety-margin
    decrease between agent a's counterfactual motions and its real one,
    both measured against neighbor b's real trajectory."""
    dg_sample = pairwise_min_distance_over_time(
        sampled_traj_a, gt_traj_b, valid_b=valid_b, d_sat=d_sat
    )  # [N]
    dg_actual = pairwise_min_distance_over_time(
        gt_traj_a.unsqueeze(0), gt_traj_b, valid_b=valid_b, d_sat=d_sat
    ).squeeze(0)
    decrease = dg_sample - dg_actual  # C(a, b, x_k), Eq. (2)
    return cvar(decrease, alpha)


def _zero(data) -> Tensor:
    return torch.zeros((), device=data["agent"]["position"].device)


# ---------------------------------------------------------------------------
# Metrics from already-drawn rollouts (no model calls)
# ---------------------------------------------------------------------------


def safety_from_rollouts(
    rollouts: Dict[str, Tensor],
    data,
    query_idx: int,
    neighbor_indices: Sequence[int],
    cfg: ResponsibilityConfig,
) -> Dict:
    """beta^safety_k(a, x_k), Eq. (3)+(5), from the queried agent's motion
    set ``rollouts`` (one free agent, drawn in the full scene)."""
    if len(neighbor_indices) == 0:
        return {"per_neighbor": {}, "value": _zero(data)}

    horizon = effective_horizon(data, cfg, rollouts["pred_traj_10hz"].shape[2])
    sampled_traj_a = rollouts["pred_traj_10hz"][:, 0, :horizon]  # [N, H, 2]
    gt_a = ground_truth_boxes(data, [query_idx], cfg.num_historical_steps, horizon)
    gt_nb = ground_truth_boxes(data, list(neighbor_indices), cfg.num_historical_steps, horizon)

    per_neighbor = {}
    for i, b in enumerate(neighbor_indices):
        per_neighbor[int(b)] = safety_responsibility_pair(
            sampled_traj_a, gt_a.pos[0], gt_nb.pos[i], gt_nb.valid[i], cfg.cvar_alpha, d_sat=cfg.d_sat
        )
    return {"per_neighbor": per_neighbor, "value": torch.stack(list(per_neighbor.values())).max()}


def _boltzmann_log_prob(reward: Tensor) -> Tensor:  # [N] -> [N]
    return F.log_softmax(reward, dim=-1)


def _reward_against_neighbors(
    ego: BoxTrajectories,
    others: BoxTrajectories,
    map_points: Optional[Tensor],
    cfg: ResponsibilityConfig,
) -> Tensor:  # [N]
    c_coll = collision_cost(ego, others, n_circles=cfg.n_collision_circles, margin=cfg.collision_margin)
    if map_points is not None and map_points.numel() > 0:
        c_dev = road_deviation_cost(ego.pos, ego.valid, map_points, cfg.drivable_half_width)
    else:
        c_dev = torch.zeros_like(c_coll)
    return -cfg.collision_weight * c_coll - c_dev


def _empty_boxes(like: Tensor, horizon: int) -> BoxTrajectories:
    return BoxTrajectories(
        pos=like.new_zeros(0, horizon, 2),
        heading=like.new_zeros(0, horizon),
        valid=torch.zeros(0, horizon, dtype=torch.bool, device=like.device),
        length=like.new_zeros(0),
        width=like.new_zeros(0),
    )


def courtesy_is_trivially_zero(data, query_idx: int, target_idx: int, cfg: ResponsibilityConfig) -> bool:
    """True when courtesy of (a, b) is not worth rolling out: no usable
    horizon, or the two agents stay further apart than
    ``courtesy_skip_distance``. For the Boltzmann proxy this is exact (the
    two rewards then differ in nothing); for the exact estimator it drops
    weak far-field influence."""
    horizon = effective_horizon(data, cfg)
    if horizon == 0:
        return True
    if cfg.courtesy_skip_distance is None:
        return False
    gt_pair = ground_truth_boxes(data, [query_idx, target_idx], cfg.num_historical_steps, horizon)
    separation = pairwise_min_distance_over_time(
        gt_pair.pos[0].unsqueeze(0), gt_pair.pos[1], valid_b=gt_pair.valid[1]
    ).squeeze(0)
    return bool(separation > cfg.courtesy_skip_distance)


def courtesy_from_rollouts(
    with_rollouts: Dict[str, Tensor],
    without_rollouts: Dict[str, Tensor],
    data,
    query_idx: int,
    target_idx: int,
    all_agent_indices: Sequence[int],
    cfg: ResponsibilityConfig,
    map_points: Optional[Tensor] = None,
) -> Tensor:  # scalar
    """beta^courtesy_k(a, b, x_k) per Sec. V-B / Eq. (7), from target b's
    rollouts drawn with (``with_rollouts``) and without a in the scene.

    The two Boltzmann distributions live on the *pooled* sample of b's
    motions and differ only in whether a's logged trajectory enters b's
    collision cost -- so this measures "how much a's motion blocks b's
    plausible options", not a change in the model's conditional
    distribution itself.
    """
    zero = _zero(data)
    horizon = effective_horizon(data, cfg, with_rollouts["pred_traj_10hz"].shape[2])
    other_with = [i for i in all_agent_indices if i != target_idx]
    if horizon == 0 or len(other_with) == 0:
        return zero
    other_without = [i for i in other_with if i != query_idx]

    pooled = {
        key: torch.cat([with_rollouts[key], without_rollouts[key]], dim=0)
        for key in ("pred_traj_10hz", "pred_head_10hz")
    }
    ego = rollout_boxes(pooled, 0, data["agent"]["shape"][target_idx], horizon)
    nb_with = ground_truth_boxes(data, other_with, cfg.num_historical_steps, horizon)
    nb_without = (
        ground_truth_boxes(data, other_without, cfg.num_historical_steps, horizon)
        if other_without
        else _empty_boxes(ego.pos, horizon)
    )

    log_p1 = _boltzmann_log_prob(_reward_against_neighbors(ego, nb_with, map_points, cfg))
    log_p2 = _boltzmann_log_prob(_reward_against_neighbors(ego, nb_without, map_points, cfg))
    return (log_p1.exp() * (log_p1 - log_p2)).sum().clamp(min=0.0)


def _check_exact_supported(cfg: "ResponsibilityConfig") -> None:
    """The two contexts of the exact estimator must differ only in the
    queried agent's presence. Under neighbor_future="simulated" the other
    agents resample their futures in each pass, which would leak into the KL."""
    if cfg.courtesy_estimator == COURTESY_EXACT and cfg.neighbor_future == NEIGHBOR_FUTURE_SIMULATED:
        raise ValueError(
            'courtesy_estimator="exact" needs neighbor_future "logged" or "hidden": '
            "under simulated the other agents resample between the two contexts"
        )


def exact_courtesy_from_logits(
    with_logits: Tensor,  # [N, S, V] target's per-step logits along its own samples, a present
    without_logits: Tensor,  # [N, S, V] the same samples replayed with a removed
    valid: Tensor,  # [N, S]
) -> Tensor:  # scalar
    """beta^courtesy_k(a, b, x_k) = KL( p(xi_b | x) || p(xi_b | x^{-a}) ), Eq. (4).

    By the chain rule the trajectory KL is the expected sum of per-step
    KLs between the two next-token distributions along prefixes drawn from
    p(. | x):

        KL(P || Q) = E_{xi ~ P} [ sum_t KL( p(. | xi_<t, x) || p(. | xi_<t, x^{-a}) ) ]

    Each inner KL is exact (a categorical over the token vocabulary), so
    averaging over the sampled prefixes gives an unbiased estimate with
    lower variance than averaging whole-trajectory log-probability ratios.
    """
    log_p = torch.log_softmax(with_logits, dim=-1)
    log_q = torch.log_softmax(without_logits, dim=-1)
    per_step = (log_p.exp() * (log_p - log_q)).sum(-1).clamp(min=0.0)  # [N, S]
    return (per_step * valid.to(per_step.dtype)).sum(-1).mean()


def _replay_without(a: int, b: int, with_rollouts: Dict[str, Tensor]) -> RolloutJob:
    """Target b replays its own sampled tokens in the scene without a."""
    tokens = with_rollouts["next_token_idx"][:, 0]  # [N, S]
    return RolloutJob(free=(), n_samples=tokens.shape[0], removed=(a,), outputs=(b,), replay={b: tokens})


# ---------------------------------------------------------------------------
# Planning: every rollout a context window needs, shared and packed
# ---------------------------------------------------------------------------


@dataclass
class ResponsibilityPlan:
    """The rollout jobs responsibility needs for one context window.

    Two sharings make this much cheaper than asking per agent and per pair:

    - An agent's *full-scene* motion set (it free, nobody removed) is the
      same distribution whether it serves as the queried agent's safety
      motion set or as a target's "with a" courtesy sample -- and the
      latter does not depend on which a is queried. So each agent gets one
      full-scene job, sized for its largest use.
    - Only "b without a" is specific to a pair.

    The cost is that an agent's safety estimate and the courtesy estimates
    that use it as a target reuse the same draws, so their Monte-Carlo
    errors are correlated; each is still an unbiased estimate on its own.
    """

    jobs: List[RolloutJob] = field(default_factory=list)
    full_job: Dict[int, int] = field(default_factory=dict)  # agent -> job index
    without_job: Dict[Tuple[int, int], int] = field(default_factory=dict)  # (a, b) -> job index
    courtesy_targets: Dict[int, List[int]] = field(default_factory=dict)  # a -> non-trivial b
    replayed_rollouts: int = 0  # replay passes run after the plan (exact courtesy)

    @property
    def n_rollouts(self) -> int:
        return sum(job.n_samples for job in self.jobs) + self.replayed_rollouts


def plan_responsibility_jobs(
    data,
    queried: Sequence[int],
    neighbors_of: Dict[int, Sequence[int]],
    cfg: ResponsibilityConfig,
) -> ResponsibilityPlan:
    plan = ResponsibilityPlan()
    full_n: Dict[int, int] = {}
    for a in queried:
        neighbours = list(neighbors_of.get(a, []))
        if not neighbours:
            plan.courtesy_targets[a] = []
            continue
        full_n[a] = max(full_n.get(a, 0), cfg.n_safety_samples)
        targets = [int(b) for b in neighbours if not courtesy_is_trivially_zero(data, a, int(b), cfg)]
        plan.courtesy_targets[a] = targets
        for b in targets:
            full_n[b] = max(full_n.get(b, 0), cfg.n_courtesy_samples)

    for agent, n in full_n.items():
        plan.full_job[agent] = len(plan.jobs)
        plan.jobs.append(RolloutJob(free=(agent,), n_samples=n))
    if cfg.courtesy_estimator == COURTESY_BOLTZMANN:
        for a, targets in plan.courtesy_targets.items():
            for b in targets:
                plan.without_job[(a, b)] = len(plan.jobs)
                plan.jobs.append(RolloutJob(free=(b,), removed=(a,), n_samples=cfg.n_courtesy_samples))
    # the exact estimator instead replays each target's full-scene samples
    # without a, which can only be planned once those samples exist
    return plan


def responsibility_for_agents(
    synthesizer: MotionSetSynthesizer,
    data,
    tokenized_map: Dict[str, Tensor],
    tokenized_agent: Dict[str, Tensor],
    queried: Sequence[int],
    neighbors_of: Dict[int, Sequence[int]],
    all_agent_indices: Sequence[int],
    cfg: ResponsibilityConfig,
    map_points: Optional[Tensor] = None,
    sampling_scheme=None,
) -> Tuple[Dict[int, Dict], ResponsibilityPlan]:
    """beta = [beta^safety, beta^courtesy] for every queried agent of one
    context window, with all the rollouts they need drawn in one packed
    pass (see :class:`ResponsibilityPlan`)."""
    if cfg.courtesy_estimator not in COURTESY_ESTIMATORS:
        raise ValueError(f"courtesy_estimator must be one of {COURTESY_ESTIMATORS}")
    _check_exact_supported(cfg)
    exact = cfg.courtesy_estimator == COURTESY_EXACT
    plan = plan_responsibility_jobs(data, queried, neighbors_of, cfg)
    common = dict(
        sampling_scheme=sampling_scheme,
        rollout_horizon=synthesizer.round_horizon(cfg.metric_horizon),
        neighbor_future=cfg.neighbor_future,
    )
    results = synthesizer.run_jobs(tokenized_map, tokenized_agent, plan.jobs, keep_logits=exact, **common)

    replayed: Dict[Tuple[int, int], Dict[str, Tensor]] = {}
    if exact:
        pairs = [(a, b) for a, targets in plan.courtesy_targets.items() for b in targets]
        jobs = [
            _replay_without(a, b, take_samples(results[plan.full_job[b]], cfg.n_courtesy_samples))
            for a, b in pairs
        ]
        replay_results = synthesizer.run_jobs(tokenized_map, tokenized_agent, jobs, keep_logits=True, **common)
        replayed = dict(zip(pairs, replay_results))
        plan.replayed_rollouts = sum(job.n_samples for job in jobs)

    zero = _zero(data)
    out: Dict[int, Dict] = {}
    for a in queried:
        neighbours = [int(b) for b in neighbors_of.get(a, [])]
        if a in plan.full_job and neighbours:
            safety = safety_from_rollouts(
                take_samples(results[plan.full_job[a]], cfg.n_safety_samples), data, a, neighbours, cfg
            )
        else:
            safety = {"per_neighbor": {}, "value": zero}

        per_neighbor = {b: zero for b in neighbours}
        for b in plan.courtesy_targets.get(a, []):
            with_b = take_samples(results[plan.full_job[b]], cfg.n_courtesy_samples)
            if exact:
                per_neighbor[b] = exact_courtesy_from_logits(
                    with_b["next_token_logits"][:, 0],
                    replayed[(a, b)]["next_token_logits"][:, 0],
                    with_b["next_token_valid"][:, 0],
                )
            else:
                per_neighbor[b] = courtesy_from_rollouts(
                    with_b,
                    results[plan.without_job[(a, b)]],
                    data,
                    a,
                    b,
                    all_agent_indices,
                    cfg,
                    map_points=map_points,
                )
        courtesy_value = torch.stack(list(per_neighbor.values())).max() if per_neighbor else zero
        courtesy = {"per_neighbor": per_neighbor, "value": courtesy_value}
        out[a] = {"beta": torch.stack([safety["value"], courtesy["value"]]), "safety": safety, "courtesy": courtesy}
    return out, plan


# ---------------------------------------------------------------------------
# Per-agent / per-pair entry points (each draws its own rollouts)
# ---------------------------------------------------------------------------


def safety_responsibility(
    synthesizer: MotionSetSynthesizer,
    data,
    tokenized_map: Dict[str, Tensor],
    tokenized_agent: Dict[str, Tensor],
    query_idx: int,
    neighbor_indices: Sequence[int],
    cfg: ResponsibilityConfig,
    sampling_scheme=None,
) -> Dict:
    """beta^safety_k(a, x_k), Eq. (5): worst case over the neighbours."""
    if len(neighbor_indices) == 0:
        return {"per_neighbor": {}, "value": _zero(data)}
    rollouts = synthesizer.sample_rollouts(
        tokenized_map,
        tokenized_agent,
        free_agent_indices=[query_idx],
        n_samples=cfg.n_safety_samples,
        sampling_scheme=sampling_scheme,
        rollout_horizon=synthesizer.round_horizon(cfg.metric_horizon),
        neighbor_future=cfg.neighbor_future,
    )
    return safety_from_rollouts(rollouts, data, query_idx, neighbor_indices, cfg)


def courtesy_responsibility_pair(
    synthesizer: MotionSetSynthesizer,
    data,
    tokenized_map: Dict[str, Tensor],
    tokenized_agent: Dict[str, Tensor],
    query_idx: int,
    target_idx: int,
    all_agent_indices: Sequence[int],
    cfg: ResponsibilityConfig,
    map_points: Optional[Tensor] = None,
    sampling_scheme=None,
) -> Tensor:
    """beta^courtesy_k(a, b, x_k); see :func:`courtesy_from_rollouts`."""
    if courtesy_is_trivially_zero(data, query_idx, target_idx, cfg):
        return _zero(data)
    if cfg.courtesy_estimator == COURTESY_EXACT:
        _check_exact_supported(cfg)
        common = dict(
            sampling_scheme=sampling_scheme,
            rollout_horizon=synthesizer.round_horizon(cfg.metric_horizon),
            neighbor_future=cfg.neighbor_future,
            keep_logits=True,
        )
        (with_b,) = synthesizer.run_jobs(
            tokenized_map, tokenized_agent,
            [RolloutJob(free=(target_idx,), n_samples=cfg.n_courtesy_samples)], **common,
        )
        (without_b,) = synthesizer.run_jobs(
            tokenized_map, tokenized_agent, [_replay_without(query_idx, target_idx, with_b)], **common
        )
        return exact_courtesy_from_logits(
            with_b["next_token_logits"][:, 0],
            without_b["next_token_logits"][:, 0],
            with_b["next_token_valid"][:, 0],
        )
    with_rollouts, without_rollouts = synthesizer.run_jobs(
        tokenized_map,
        tokenized_agent,
        [
            RolloutJob(free=(target_idx,), n_samples=cfg.n_courtesy_samples),
            RolloutJob(free=(target_idx,), removed=(query_idx,), n_samples=cfg.n_courtesy_samples),
        ],
        sampling_scheme=sampling_scheme,
        rollout_horizon=synthesizer.round_horizon(cfg.metric_horizon),
        neighbor_future=cfg.neighbor_future,
    )
    return courtesy_from_rollouts(
        with_rollouts, without_rollouts, data, query_idx, target_idx, all_agent_indices, cfg, map_points
    )


def courtesy_responsibility(
    synthesizer: MotionSetSynthesizer,
    data,
    tokenized_map: Dict[str, Tensor],
    tokenized_agent: Dict[str, Tensor],
    query_idx: int,
    neighbor_indices: Sequence[int],
    all_agent_indices: Sequence[int],
    cfg: ResponsibilityConfig,
    map_points: Optional[Tensor] = None,
    sampling_scheme=None,
) -> Dict:
    """beta^courtesy_k(a, x_k), Eq. (5): worst case over the neighbours."""
    if len(neighbor_indices) == 0:
        return {"per_neighbor": {}, "value": _zero(data)}
    per_neighbor = {
        int(b): courtesy_responsibility_pair(
            synthesizer, data, tokenized_map, tokenized_agent, query_idx, int(b),
            all_agent_indices, cfg, map_points=map_points, sampling_scheme=sampling_scheme,
        )
        for b in neighbor_indices
    }
    return {"per_neighbor": per_neighbor, "value": torch.stack(list(per_neighbor.values())).max()}


def compute_agent_responsibility(
    synthesizer: MotionSetSynthesizer,
    data,
    tokenized_map: Dict[str, Tensor],
    tokenized_agent: Dict[str, Tensor],
    query_idx: int,
    neighbor_indices: Sequence[int],
    all_agent_indices: Sequence[int],
    cfg: ResponsibilityConfig,
    map_points: Optional[Tensor] = None,
    sampling_scheme=None,
) -> Dict[str, Tensor]:
    """beta_k := [beta^safety_k, beta^courtesy_k]^T (Sec. IV) for one queried
    agent, drawn through the same packed plan as a whole window."""
    results, _ = responsibility_for_agents(
        synthesizer,
        data,
        tokenized_map,
        tokenized_agent,
        [query_idx],
        {query_idx: list(neighbor_indices)},
        all_agent_indices,
        cfg,
        map_points=map_points,
        sampling_scheme=sampling_scheme,
    )
    return results[query_idx]

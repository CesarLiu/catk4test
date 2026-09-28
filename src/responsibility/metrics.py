"""Safety and courtesy responsibility (paper Sec. III-B, Eq. 2-5, and the
tractable Boltzmann approximation to courtesy responsibility from Sec. V-B,
Eq. 7, which we use in place of the intractable exact KL of Eq. 4).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from src.responsibility.geometry import (
    BoxTrajectories,
    collision_cost,
    pairwise_min_distance_over_time,
    road_deviation_cost,
)
from src.responsibility.motion_set import NEIGHBOR_FUTURE_LOGGED, MotionSetSynthesizer
from src.responsibility.risk import cvar


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
    # Courtesy is provably zero when the queried agent never enters the
    # collision term of the target's reward. Skipping those pairs before
    # rolling out saves the bulk of the compute; set to None to disable.
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
    """beta^safety_k(a, x_k), Eq. (5): worst-case (max) safety responsibility
    of ``query_idx`` over all its neighbors."""
    if len(neighbor_indices) == 0:
        return {"per_neighbor": {}, "value": torch.tensor(0.0)}

    rollouts = synthesizer.sample_rollouts(
        tokenized_map,
        tokenized_agent,
        free_agent_indices=[query_idx],
        n_samples=cfg.n_safety_samples,
        sampling_scheme=sampling_scheme,
        rollout_horizon=synthesizer.round_horizon(cfg.metric_horizon),
        neighbor_future=cfg.neighbor_future,
    )
    horizon = effective_horizon(data, cfg, rollouts["pred_traj_10hz"].shape[2])
    sampled_traj_a = rollouts["pred_traj_10hz"][:, 0, :horizon]  # [N, H, 2]

    gt_a = ground_truth_boxes(data, [query_idx], cfg.num_historical_steps, horizon)
    gt_nb = ground_truth_boxes(data, list(neighbor_indices), cfg.num_historical_steps, horizon)

    per_neighbor = {}
    for i, b in enumerate(neighbor_indices):
        per_neighbor[int(b)] = safety_responsibility_pair(
            sampled_traj_a,
            gt_a.pos[0],
            gt_nb.pos[i],
            gt_nb.valid[i],
            cfg.cvar_alpha,
            d_sat=cfg.d_sat,
        )

    value = torch.stack(list(per_neighbor.values())).max()
    return {"per_neighbor": per_neighbor, "value": value}


def _boltzmann_log_prob(reward: Tensor) -> Tensor:  # [N] -> [N]
    return F.log_softmax(reward, dim=-1)


def _reward_against_neighbors(
    ego: BoxTrajectories,
    others: BoxTrajectories,
    map_points: Optional[Tensor],
    cfg: ResponsibilityConfig,
) -> Tensor:  # [N]
    c_coll = collision_cost(
        ego, others, n_circles=cfg.n_collision_circles, margin=cfg.collision_margin
    )
    if map_points is not None and map_points.numel() > 0:
        c_dev = road_deviation_cost(ego.pos, ego.valid, map_points, cfg.drivable_half_width)
    else:
        c_dev = torch.zeros_like(c_coll)
    return -cfg.collision_weight * c_coll - c_dev


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
) -> Tensor:  # scalar
    """beta^courtesy_k(a, b, x_k), approximated per Sec. V-B / Eq. (7): the
    KL divergence between two Boltzmann distributions over a pooled sample
    of target agent b's motions, induced by a reward function that does
    (P1) / does not (P2) include queried agent a among b's neighbors.

    Note what this approximation can and cannot see: the only term that
    differs between the two rewards is b's collision cost against a, so it
    measures "how much a's motion blocks b's plausible options", not a
    change in the predictor's conditional distribution. See
    ``exact_courtesy_responsibility_pair`` for the likelihood-based
    alternative this repo's tokenized model also makes available.
    """
    zero = torch.zeros((), device=data["agent"]["position"].device)
    horizon = effective_horizon(data, cfg)
    if horizon == 0:
        return zero

    if cfg.courtesy_skip_distance is not None:
        gt_pair = ground_truth_boxes(data, [query_idx, target_idx], cfg.num_historical_steps, horizon)
        separation = pairwise_min_distance_over_time(
            gt_pair.pos[0].unsqueeze(0),
            gt_pair.pos[1],
            valid_b=gt_pair.valid[1],
        ).squeeze(0)
        if separation > cfg.courtesy_skip_distance:
            return zero

    rollouts_with = synthesizer.sample_rollouts(
        tokenized_map,
        tokenized_agent,
        free_agent_indices=[target_idx],
        n_samples=cfg.n_courtesy_samples,
        remove_agent_indices=None,
        sampling_scheme=sampling_scheme,
        rollout_horizon=synthesizer.round_horizon(cfg.metric_horizon),
        neighbor_future=cfg.neighbor_future,
    )
    rollouts_without = synthesizer.sample_rollouts(
        tokenized_map,
        tokenized_agent,
        free_agent_indices=[target_idx],
        n_samples=cfg.n_courtesy_samples,
        remove_agent_indices=[query_idx],
        sampling_scheme=sampling_scheme,
        rollout_horizon=synthesizer.round_horizon(cfg.metric_horizon),
        neighbor_future=cfg.neighbor_future,
    )

    horizon = effective_horizon(data, cfg, rollouts_with["pred_traj_10hz"].shape[2])
    pooled = {
        "pred_traj_10hz": torch.cat(
            [rollouts_with["pred_traj_10hz"], rollouts_without["pred_traj_10hz"]], dim=0
        ),
        "pred_head_10hz": torch.cat(
            [rollouts_with["pred_head_10hz"], rollouts_without["pred_head_10hz"]], dim=0
        ),
    }
    ego = rollout_boxes(pooled, 0, data["agent"]["shape"][target_idx], horizon)

    other_with = [i for i in all_agent_indices if i != target_idx]
    other_without = [i for i in other_with if i != query_idx]
    if len(other_with) == 0:
        return zero

    nb_with = ground_truth_boxes(data, other_with, cfg.num_historical_steps, horizon)
    nb_without = (
        ground_truth_boxes(data, other_without, cfg.num_historical_steps, horizon)
        if len(other_without) > 0
        else BoxTrajectories(
            pos=ego.pos.new_zeros(0, horizon, 2),
            heading=ego.pos.new_zeros(0, horizon),
            valid=torch.zeros(0, horizon, dtype=torch.bool, device=ego.pos.device),
            length=ego.pos.new_zeros(0),
            width=ego.pos.new_zeros(0),
        )
    )

    reward_with = _reward_against_neighbors(ego, nb_with, map_points, cfg)
    reward_without = _reward_against_neighbors(ego, nb_without, map_points, cfg)

    log_p1 = _boltzmann_log_prob(reward_with)
    log_p2 = _boltzmann_log_prob(reward_without)
    kl = (log_p1.exp() * (log_p1 - log_p2)).sum()
    return kl.clamp(min=0.0)


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
    """beta^courtesy_k(a, x_k), Eq. (5): worst-case courtesy responsibility
    of ``query_idx`` over all its neighbors."""
    if len(neighbor_indices) == 0:
        return {"per_neighbor": {}, "value": torch.tensor(0.0)}

    per_neighbor = {}
    for b in neighbor_indices:
        per_neighbor[int(b)] = courtesy_responsibility_pair(
            synthesizer,
            data,
            tokenized_map,
            tokenized_agent,
            query_idx,
            int(b),
            all_agent_indices,
            cfg,
            map_points=map_points,
            sampling_scheme=sampling_scheme,
        )
    value = torch.stack(list(per_neighbor.values())).max()
    return {"per_neighbor": per_neighbor, "value": value}


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
    """Convenience wrapper computing beta = [beta^safety, beta^courtesy] for
    one queried agent at the current context, matching the paper's notation
    (Sec. IV): beta_k := [beta^safety_k, beta^courtesy_k]^T."""
    safety = safety_responsibility(
        synthesizer,
        data,
        tokenized_map,
        tokenized_agent,
        query_idx,
        neighbor_indices,
        cfg,
        sampling_scheme=sampling_scheme,
    )
    courtesy = courtesy_responsibility(
        synthesizer,
        data,
        tokenized_map,
        tokenized_agent,
        query_idx,
        neighbor_indices,
        all_agent_indices,
        cfg,
        map_points=map_points,
        sampling_scheme=sampling_scheme,
    )
    beta = torch.stack([safety["value"], courtesy["value"]])
    return {"beta": beta, "safety": safety, "courtesy": courtesy}

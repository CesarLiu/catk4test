"""Geometric primitives for counterfactual responsibility.

All functions are pure tensor math (no autograd needed downstream, but
gradients are not blocked either) so they work equally on cached CPU
tensors from an offline analysis script or on GPU tensors during an
online prediction pass.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass
class BoxTrajectories:
    """A set of co-temporal trajectories of oriented boxes.

    pos:     [N, T, 2]   center positions
    heading: [N, T]      heading angles (rad)
    valid:   [N, T]      per-step validity
    length:  [N]         box length (longitudinal extent)
    width:   [N]         box width (lateral extent)
    """

    pos: Tensor
    heading: Tensor
    valid: Tensor
    length: Tensor
    width: Tensor

    def __len__(self) -> int:
        return self.pos.shape[0]

    @property
    def horizon(self) -> int:
        return self.pos.shape[1]

    def truncate(self, horizon: int) -> "BoxTrajectories":
        if horizon >= self.horizon:
            return self
        return BoxTrajectories(
            pos=self.pos[:, :horizon],
            heading=self.heading[:, :horizon],
            valid=self.valid[:, :horizon],
            length=self.length,
            width=self.width,
        )


def saturate(distance: Tensor, d_sat: Optional[float]) -> Tensor:
    """Caps a distance at ``d_sat``.

    The paper leaves D_g unsaturated, but then reports a safety-responsibility
    axis that tops out around 1.2 m with most mass at zero (Fig. 2/3), which an
    unsaturated min-distance difference cannot produce for far-apart pairs: any
    alternative motion trivially "gains" many meters against a distant
    neighbor. Saturating at an interaction range keeps the metric about
    *interaction* rather than about how far away the other car happened to be.
    """
    if d_sat is None:
        return distance
    return distance.clamp(max=d_sat)


def safety_margin(
    traj: Tensor,  # [..., T, 2]
    ref_traj: Tensor,  # [T, 2]
    ref_valid: Optional[Tensor] = None,  # [T]
    d_sat: Optional[float] = None,
) -> Tensor:  # [...]
    """D_g(xi, xi') := min_t || xi_t - xi'_t ||_2, Eq. (2) in the paper.

    ``traj`` may carry arbitrary leading batch dims (e.g. a set of N
    sampled counterfactual rollouts); ``ref_traj`` is a single reference
    trajectory (e.g. the ground-truth log of a neighbor).
    """
    dist = torch.linalg.norm(traj - ref_traj, dim=-1)  # [..., T]
    if ref_valid is not None:
        dist = dist.masked_fill(~ref_valid, torch.inf)
    min_dist = dist.min(dim=-1).values
    return saturate(torch.nan_to_num(min_dist, posinf=0.0), d_sat)


def pairwise_min_distance_over_time(
    traj_a: Tensor,  # [..., T, 2]
    traj_b: Tensor,  # [T, 2]
    valid_a: Optional[Tensor] = None,  # [..., T]
    valid_b: Optional[Tensor] = None,  # [T]
    d_sat: Optional[float] = None,
) -> Tensor:  # [...]
    """Closest approach between two co-temporal trajectories.

    Same as :func:`safety_margin` but also allows masking out invalid
    steps of ``traj_a`` (used e.g. for a rollout that only partially
    overlaps the reference agent's valid horizon).
    """
    dist = torch.linalg.norm(traj_a - traj_b, dim=-1)  # [..., T]
    valid = torch.ones_like(dist, dtype=torch.bool)
    if valid_a is not None:
        valid = valid & valid_a
    if valid_b is not None:
        valid = valid & valid_b
    dist = dist.masked_fill(~valid, torch.inf)
    min_dist = dist.min(dim=-1).values
    return saturate(torch.nan_to_num(min_dist, posinf=0.0), d_sat)


def circle_decomposition(
    length: Tensor,  # [N]
    width: Tensor,  # [N]
    n_circles: int,
) -> Tuple[Tensor, Tensor]:  # offsets [N, n_circles], radius [N]
    """Covers each box with ``n_circles`` equal circles along its long axis.

    Centers sit at the midpoints of n equal longitudinal slices and the
    radius is the half-diagonal of one slice, so the union of circles
    contains the box exactly. A single circle (n=1) inscribing the whole
    box is far too conservative for traffic: a 4.8x2.0 m vehicle would get
    r=2.4 m, making any two cars in adjacent lanes (~3.5 m apart) register
    a permanent collision.
    """
    steps = torch.arange(n_circles, dtype=length.dtype, device=length.device)
    frac = (2.0 * steps + 1.0) / (2.0 * n_circles) - 0.5  # [n_circles]
    offsets = length.unsqueeze(-1) * frac  # [N, n_circles]
    radius = torch.sqrt((length / (2.0 * n_circles)) ** 2 + (width / 2.0) ** 2)  # [N]
    return offsets, radius


def circle_centers(
    pos: Tensor,  # [N, T, 2]
    heading: Tensor,  # [N, T]
    offsets: Tensor,  # [N, C]
) -> Tensor:  # [N, T, C, 2]
    direction = torch.stack([heading.cos(), heading.sin()], dim=-1)  # [N, T, 2]
    return pos.unsqueeze(-2) + direction.unsqueeze(-2) * offsets[:, None, :, None]


def collision_cost(
    ego: BoxTrajectories,
    others: BoxTrajectories,
    n_circles: int = 3,
    margin: float = 0.0,
) -> Tensor:  # [N]
    """Soft collision cost of each ego trajectory against a fixed set of
    other agents' trajectories.

    Both parties are covered by circles (see :func:`circle_decomposition`);
    at each timestep the deepest penetration over all circle pairs is
    taken, then summed over timesteps and other agents. This is the
    ``c_coll`` term of the paper's Sec. V-B reward ``-10*c_coll - c_dev``.
    """
    return collision_cost_matrix(ego, others, n_circles=n_circles, margin=margin).sum(dim=-1)


def collision_cost_matrix(
    ego: BoxTrajectories,
    others: BoxTrajectories,
    n_circles: int = 3,
    margin: float = 0.0,
) -> Tensor:  # [N, M]
    """:func:`collision_cost` of every ego trajectory against every other
    agent separately, summed over time; mask entries (e.g. an agent against
    itself) before summing over the others."""
    n_ego = len(ego)
    if len(others) == 0 or ego.horizon == 0:
        return torch.zeros(n_ego, len(others), device=ego.pos.device, dtype=ego.pos.dtype)

    ego_offsets, ego_radius = circle_decomposition(ego.length, ego.width, n_circles)
    other_offsets, other_radius = circle_decomposition(others.length, others.width, n_circles)

    ego_c = circle_centers(ego.pos, ego.heading, ego_offsets)  # [N, T, C, 2]
    other_c = circle_centers(others.pos, others.heading, other_offsets)  # [M, T, C, 2]

    # [N, 1, T, C, 1, 2] - [1, M, T, 1, C, 2] -> [N, M, T, C, C]
    diff = ego_c[:, None, :, :, None, :] - other_c[None, :, :, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1)

    combined = ego_radius[:, None] + other_radius[None, :] + margin  # [N, M]
    penetration = (combined[:, :, None, None, None] - dist).clamp(min=0.0)
    penetration = penetration.amax(dim=(-2, -1))  # deepest circle pair -> [N, M, T]

    valid = ego.valid[:, None, :] & others.valid[None, :, :]  # [N, M, T]
    return penetration.masked_fill(~valid, 0.0).sum(dim=-1)


def road_deviation_cost(
    traj: Tensor,  # [..., T, 2]
    traj_valid: Tensor,  # [..., T]
    map_points: Tensor,  # [M, 2]
    drivable_half_width: float = 3.0,
    chunk_size: int = 4096,
) -> Tensor:  # [...]
    """Penalize trajectory points far from the nearest drivable map point.

    This is an approximation of a signed-distance-to-drivable-area cost
    (as e.g. used in BITS): the exact drivable polygon is not reified in
    this repo's cached WOMD map format, so we instead penalize distance
    beyond ``drivable_half_width`` to the nearest map polyline sample
    point. Use ``scripts/responsibility/inspect_map_types.py`` to check
    which ``pt_token.pl_type`` values correspond to lanes/road edges in
    your cached data and pre-filter ``map_points`` accordingly for a
    tighter approximation.
    """
    if map_points.numel() == 0:
        return torch.zeros(traj.shape[:-2], device=traj.device, dtype=traj.dtype)

    flat_traj = traj.reshape(-1, traj.shape[-1])  # [N, 2]
    min_dist = torch.full(
        (flat_traj.shape[0],), torch.inf, device=traj.device, dtype=traj.dtype
    )
    for start in range(0, flat_traj.shape[0], chunk_size):
        chunk = flat_traj[start : start + chunk_size]  # [c, 2]
        d = torch.cdist(chunk, map_points)  # [c, M]
        min_dist[start : start + chunk_size] = d.min(dim=-1).values

    dev = (min_dist - drivable_half_width).clamp(min=0.0)
    dev = dev.view(traj.shape[:-1])  # [..., T]
    dev = dev.masked_fill(~traj_valid, 0.0)
    return dev.sum(dim=-1)


def final_displacement_progress(
    traj: Tensor,  # [..., T, 2]
    traj_valid: Tensor,  # [..., T]
    origin: Tensor,  # [..., 2] or [2]
) -> Tensor:  # [...]
    """Net displacement from ``origin`` to the last valid point of ``traj``."""
    T = traj.shape[-2]
    idx_range = torch.arange(T, device=traj.device).expand(traj_valid.shape)
    last_valid_idx = (idx_range * traj_valid).amax(dim=-1)  # [...]
    last_pos = torch.gather(
        traj, -2, last_valid_idx[..., None, None].expand(*traj_valid.shape[:-1], 1, 2)
    ).squeeze(-2)
    any_valid = traj_valid.any(dim=-1)
    disp = torch.linalg.norm(last_pos - origin, dim=-1)
    return disp.masked_fill(~any_valid, 0.0)

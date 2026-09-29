"""Builds a responsibility-augmented dataset (paper Sec. IV-B) from cached
WOMD scenarios: per-agent responsibility observations, HMM-based level
labels, and per-level (real vs. auxiliary) trajectory feature sets used to
train the reward networks in :mod:`src.responsibility.reward_model`.

Responsibility *sequences*. The paper re-evaluates Eq. (3)/(4) at every
0.1s step of nuScenes' 20s scenes, and the HMM's transition matrix (Sec.
IV-A) only means anything if it is fitted on such sequences. This repo's
cached WOMD scenarios are short (~9.1s), non-overlapping clips, so we
instead sweep the *current step* within a clip: :func:`window_steps`
picks k = k0, k0+stride, ..., and each one is evaluated under
:func:`~src.responsibility.motion_set.shifted_context`, which re-anchors
the clip so that k is the present. One clip therefore yields a genuine
multi-step beta sequence per agent.

Each beta at k needs ``metric_horizon`` steps of *logged future* to be
computable, so it is only known at k + metric_horizon. Anything that
consumes a level to make a prediction at time k_predict must therefore
use windows ending at or before k_predict -- see
:func:`causal_window_steps`, which enforces exactly that and is what the
online selection path uses to avoid conditioning on the future it is
being scored against.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from src.responsibility.hmm import GaussianHMM
from src.responsibility.interaction import (
    QUERY_SDC,
    select_interacting_neighbours,
    select_query_agents,
)
from src.responsibility.metrics import (
    DIRECTION_BOTH,
    DIRECTION_INCOMING,
    ResponsibilityConfig,
    responsibility_for_agents,
    effective_horizon,
    ground_truth_boxes,
    others_future_boxes,
    rollout_boxes,
)
from src.responsibility.motion_set import (
    NEIGHBOR_FUTURE_LOGGED,
    MotionSetSynthesizer,
    RolloutJob,
    find_neighbors,
    shifted_context,
)
from src.responsibility.reward_model import extract_trajectory_features, mean_token_log_prob


SCOPE_NEIGHBOURS = "neighbours"  # the agent toward all its interacting neighbours
SCOPE_TOWARD_SDC = "toward_sdc"  # the agent toward the self-driving car only


@dataclass
class ResponsibilityObservation:
    scenario_id: str
    agent_id: int
    agent_idx: int
    k: int
    beta: Tensor  # [2]: (safety, courtesy)
    level: Optional[int] = None
    # which neighbour set beta is a max over; sequences never mix scopes
    scope: str = SCOPE_NEIGHBOURS
    # neighbour agent id -> (safety, courtesy) before the max of Eq. (5)
    per_neighbour: Optional[Dict[int, Tuple[float, float]]] = None
    # neighbour agent id -> interaction evidence (min_gap, pet, ttc, influence)
    interaction: Optional[Dict[int, Dict[str, float]]] = None


# Map encoding of src/data_preprocess.py: pt_token.pl_type indexes
# ["lane", "road_edge", "road_line", "crosswalk"]; pt_token.type is the
# point type, of which 0 (freeway) and 1 (surface street) are vehicle lanes
# (3 is a bike lane; 4-5 road edges, 6-8 road lines, 9 crosswalks).
LANE_POLYGON = 0
VEHICLE_LANE_POINT_TYPES = (0, 1)


def get_map_points(data, vehicle_lanes_only: bool = True) -> Optional[Tensor]:
    """Reference points for the road-deviation cost: every sample point
    (three per 5 m segment) of the vehicle-lane centerlines, so a vehicle
    driving in a lane sits within about half a lane width of one. With
    ``vehicle_lanes_only=False``, or when a map has no such lanes, every map
    point is used (crosswalks, stop lines and road edges included)."""
    try:
        traj_pos = data["map_save"]["traj_pos"]  # [n_pl, 3, 2]
    except (KeyError, IndexError):
        return None
    if vehicle_lanes_only:
        try:
            pl_type, pt_type = data["pt_token"]["pl_type"], data["pt_token"]["type"]
        except (KeyError, IndexError):
            pl_type = pt_type = None
        if pl_type is not None:
            lanes = pl_type == LANE_POLYGON
            vehicle = torch.zeros_like(lanes)
            for t in VEHICLE_LANE_POINT_TYPES:
                vehicle |= pt_type == t
            keep = lanes & vehicle
            if bool(keep.any()):
                traj_pos = traj_pos[keep]
    return traj_pos.reshape(-1, 2)


def last_usable_window_step(
    n_step_total: int, cfg: ResponsibilityConfig, token_shift: int, rollout_horizon: int
) -> int:
    """Largest current step k for which a shifted clip still has enough
    tokenized future for the model to roll out ``rollout_horizon`` steps.

    The tokenizer lays tokens on a ``token_shift`` grid and the rollout
    consumes ``step_current + rollout_horizon/token_shift`` of them, so a
    window too close to the end of the clip would index past the token
    arrays inside the decoder.
    """
    step_current_2hz = (cfg.num_historical_steps - 1) // token_shift
    needed = token_shift * (step_current_2hz + rollout_horizon // token_shift) + 1
    return n_step_total + cfg.num_historical_steps - 1 - needed


def window_steps(
    data,
    cfg: ResponsibilityConfig,
    synthesizer: MotionSetSynthesizer,
    stride: int,
    first_step: Optional[int] = None,
    last_step: Optional[int] = None,
) -> List[int]:
    """Current steps at which responsibility is evaluated within one clip."""
    n_step_total = data["agent"]["position"].shape[1]
    rollout_horizon = synthesizer.round_horizon(cfg.metric_horizon)
    k0 = cfg.num_historical_steps - 1 if first_step is None else first_step
    k_last = last_usable_window_step(n_step_total, cfg, synthesizer.token_shift, rollout_horizon)
    if last_step is not None:
        k_last = min(k_last, last_step)
    return list(range(k0, k_last + 1, max(1, stride)))


def causal_window_steps(
    data,
    cfg: ResponsibilityConfig,
    synthesizer: MotionSetSynthesizer,
    predict_step: int,
    stride: int,
) -> List[int]:
    """Window steps whose responsibility is already *known* at ``predict_step``.

    beta at k is computed from the logged future over [k+1, k+metric_horizon],
    so it is only observable from k + metric_horizon onwards. Using a level
    derived from a later window to make a prediction at ``predict_step``
    would condition the prediction on the ground truth it is scored
    against; this is the guard against that.
    """
    return window_steps(
        data,
        cfg,
        synthesizer,
        stride,
        last_step=predict_step - cfg.metric_horizon,
    )


def compute_scenario_responsibility(
    synthesizer: MotionSetSynthesizer,
    data,
    cfg: ResponsibilityConfig,
    agent_type_filter: Optional[int] = 0,
    max_agents: Optional[int] = None,
    steps: Optional[Sequence[int]] = None,
    agents: Optional[Sequence[int]] = None,
    stats: Optional[Dict[str, int]] = None,
) -> List[ResponsibilityObservation]:
    """Responsibility observations (Eq. 3/4/5) for every agent matching
    ``agent_type_filter`` (default 0 = vehicle, the paper's vehicle-vehicle
    scope; ``None`` for all types), at each requested current step.

    ``steps`` defaults to the clip's native current step; pass the output
    of :func:`window_steps` to get a per-agent sequence instead. ``agents``
    restricts the query to those agent indices (still subject to being
    observed at each step).
    """
    # data comes from a batch_size=1 DataLoader, so PyG collates
    # "scenario_id" (a per-graph, non-tensor attribute) into a length-1 list.
    # Callers that already know the scenario_id (e.g. compute_dataset_responsibility)
    # overwrite this afterward; this is only a fallback for standalone use.
    if "scenario_id" in data:
        raw_scenario_id = data["scenario_id"]
        scenario_id = str(raw_scenario_id[0]) if isinstance(raw_scenario_id, (list, tuple)) else str(raw_scenario_id)
    else:
        scenario_id = "unknown"

    if steps is None:
        steps = [cfg.num_historical_steps - 1]

    observations: List[ResponsibilityObservation] = []
    for k in steps:
        with shifted_context(data, k, cfg.num_historical_steps):
            observations.extend(
                _responsibility_at_current_step(
                    synthesizer, data, cfg, scenario_id, k, agent_type_filter, max_agents, agents, stats
                )
            )
    return observations


def _responsibility_at_current_step(
    synthesizer: MotionSetSynthesizer,
    data,
    cfg: ResponsibilityConfig,
    scenario_id: str,
    k: int,
    agent_type_filter: Optional[int],
    max_agents: Optional[int],
    agents: Optional[Sequence[int]] = None,
    stats: Optional[Dict[str, int]] = None,
) -> List[ResponsibilityObservation]:
    """``data`` must already be anchored so that ``cfg.num_historical_steps - 1``
    is the current step (see ``shifted_context``); ``k`` is only carried
    through to the observation for bookkeeping."""
    tokenized_map, tokenized_agent = synthesizer.tokenize(data)
    n_agent = data["agent"]["position"].shape[0]
    at_step = cfg.num_historical_steps - 1
    map_points = get_map_points(data)
    all_idx = list(range(n_agent))

    if agents is not None:
        observed = data["agent"]["valid_mask"][:, at_step]
        queried = [int(a) for a in agents if bool(observed[int(a)])]
    else:
        queried = select_query_agents(data, at_step, cfg.query_mode, agent_type_filter)
    if max_agents is not None:
        queried = queried[:max_agents]
    selection = select_interacting_neighbours(
        synthesizer,
        data,
        queried,
        cfg.num_historical_steps,
        effective_horizon(data, cfg),
        cfg.interaction,
        tokenized_map=tokenized_map,
        tokenized_agent=tokenized_agent,
        radius=cfg.neighbor_radius,
    )
    neighbours_of = dict(selection.neighbours)
    scope_of = {a: SCOPE_NEIGHBOURS for a in queried}
    interaction_of = {a: selection.diagnostics[a] for a in queried}

    if cfg.query_mode == QUERY_SDC and cfg.query_direction in (DIRECTION_INCOMING, DIRECTION_BOTH):
        # the reverse question: how responsible was each interacting agent
        # toward the self-driving car
        for ego in list(queried):
            for b in selection.neighbours[ego]:
                if b not in neighbours_of:
                    neighbours_of[b] = [ego]
                    scope_of[b] = SCOPE_TOWARD_SDC
                    interaction_of[b] = {ego: selection.diagnostics[ego][b]}
        if cfg.query_direction == DIRECTION_INCOMING:
            for ego in queried:
                del neighbours_of[ego], scope_of[ego], interaction_of[ego]

    agents = list(neighbours_of)
    # one packed pass for every agent of this window (see ResponsibilityPlan)
    results, plan = responsibility_for_agents(
        synthesizer, data, tokenized_map, tokenized_agent, agents, neighbours_of, all_idx, cfg,
        map_points=map_points,
    )
    if stats is not None:
        stats["windows"] = stats.get("windows", 0) + 1
        stats["rollouts"] = stats.get("rollouts", 0) + plan.n_rollouts
    agent_id = data["agent"]["id"]
    observations = []
    for a in agents:
        r = results[a]
        per_neighbour = {
            int(agent_id[b]): (float(r["safety"]["per_neighbor"].get(b, 0.0)),
                               float(r["courtesy"]["per_neighbor"].get(b, 0.0)))
            for b in neighbours_of[a]
        }
        observations.append(
            ResponsibilityObservation(
                scenario_id=str(scenario_id),
                agent_id=int(agent_id[a]),
                agent_idx=a,
                k=k,
                beta=r["beta"].detach().cpu(),
                scope=scope_of[a],
                per_neighbour=per_neighbour,
                interaction={int(agent_id[b]): v for b, v in interaction_of[a].items()},
            )
        )
    return observations


def group_sequences(
    observations: Sequence[ResponsibilityObservation],
) -> Dict[Tuple[str, int, str], List[ResponsibilityObservation]]:
    groups: Dict[Tuple[str, int, str], List[ResponsibilityObservation]] = {}
    for obs in observations:
        key = (obs.scenario_id, obs.agent_id, getattr(obs, "scope", SCOPE_NEIGHBOURS))
        groups.setdefault(key, []).append(obs)
    for key in groups:
        groups[key].sort(key=lambda o: o.k)
    return groups


def sequences_as_arrays(
    grouped: Dict[Tuple[str, int], List[ResponsibilityObservation]]
) -> List[np.ndarray]:
    return [np.stack([o.beta.numpy() for o in obs_list]) for obs_list in grouped.values()]


def assign_levels(
    grouped: Dict[Tuple[str, int], List[ResponsibilityObservation]], hmm: GaussianHMM
) -> None:
    """In-place: sets ``.level`` on every observation via the Bayes filter
    (Eq. 6), applied along each agent's own sequence of observations."""
    for obs_list in grouped.values():
        seq = np.stack([o.beta.numpy() for o in obs_list])
        levels = hmm.predict_level(seq)
        for obs, level in zip(obs_list, levels):
            obs.level = int(level)


@dataclass
class LevelFeatureSet:
    real: Tensor  # [N, 4]
    fake: Tensor  # [N * n_aux, 4]


def build_reward_training_features(
    synthesizer: MotionSetSynthesizer,
    data_by_scenario: Dict[str, "object"],
    observations: Sequence[ResponsibilityObservation],
    cfg: ResponsibilityConfig,
    n_aux: int = 24,
    sampling_scheme=None,
) -> Dict[int, LevelFeatureSet]:
    """For every leveled observation, extracts the real trajectory's feature
    vector and ``n_aux`` auxiliary (freely sampled) trajectories' feature
    vectors, bucketed by responsibility level (Sec. IV-B/V-C)."""
    return stream_reward_training_features(
        synthesizer, data_by_scenario.items(), observations, cfg, n_aux=n_aux, sampling_scheme=sampling_scheme
    )


def stream_reward_training_features(
    synthesizer: MotionSetSynthesizer,
    scenarios: Iterable[Tuple[str, "object"]],
    observations: Sequence[ResponsibilityObservation],
    cfg: ResponsibilityConfig,
    n_aux: int = 24,
    sampling_scheme=None,
) -> Dict[int, LevelFeatureSet]:
    """Same as :func:`build_reward_training_features`, but consuming
    ``(scenario_id, data)`` pairs one at a time, so only the scenario being
    processed is resident -- holding every scenario on the GPU at once does
    not scale to a real dataset. Scenarios without leveled observations are
    skipped; the features themselves are kept on CPU."""
    by_scenario: Dict[str, List[ResponsibilityObservation]] = {}
    for obs in observations:
        if obs.level is not None:
            by_scenario.setdefault(obs.scenario_id, []).append(obs)

    buckets: Dict[int, Tuple[List[Tensor], List[Tensor]]] = {}
    for scenario_id, scenario in scenarios:
        for obs in by_scenario.pop(scenario_id, []):
            with shifted_context(scenario, obs.k, cfg.num_historical_steps) as data:
                real_feat, fake_feat = _features_for_observation(
                    synthesizer, data, obs, cfg, n_aux, sampling_scheme
                )
            real_list, fake_list = buckets.setdefault(obs.level, ([], []))
            real_list.append(real_feat.detach().cpu())
            fake_list.append(fake_feat.detach().cpu())
        if not by_scenario:
            break

    return {
        level: LevelFeatureSet(real=torch.stack(real_list), fake=torch.cat(fake_list, dim=0))
        for level, (real_list, fake_list) in buckets.items()
        if len(real_list) > 0
    }


def _features_for_observation(
    synthesizer: MotionSetSynthesizer,
    data,
    obs: ResponsibilityObservation,
    cfg: ResponsibilityConfig,
    n_aux: int,
    sampling_scheme,
) -> Tuple[Tensor, Tensor]:
    """Feature vectors of one logged trajectory and of ``n_aux`` auxiliary
    ones sampled from the model in the same context."""
    tokenized_map, tokenized_agent = synthesizer.tokenize(data)
    n_agent = data["agent"]["position"].shape[0]
    all_idx = [i for i in range(n_agent) if i != obs.agent_idx]
    map_points = get_map_points(data)
    horizon_arg = synthesizer.round_horizon(cfg.metric_horizon)

    aux_job = RolloutJob(free=(obs.agent_idx,), n_samples=n_aux)
    # The logged trajectory's own model log-probability, from a pass in
    # which *every* agent replays ground truth (i.e. teacher forcing).
    # Leaving this at 0 for real trajectories and negative for sampled
    # ones would hand the discriminator a trivial shortcut feature.
    gt_job = RolloutJob(free=(), n_samples=1, outputs=(obs.agent_idx,))
    common = dict(sampling_scheme=sampling_scheme, rollout_horizon=horizon_arg)
    if cfg.neighbor_future == NEIGHBOR_FUTURE_LOGGED:
        rollouts, gt_scored = synthesizer.run_jobs(
            tokenized_map, tokenized_agent, [aux_job, gt_job], neighbor_future=NEIGHBOR_FUTURE_LOGGED, **common
        )
    else:
        (rollouts,) = synthesizer.run_jobs(
            tokenized_map, tokenized_agent, [aux_job], neighbor_future=cfg.neighbor_future, **common
        )
        # scoring the log must stay teacher-forced whatever the sampling mode
        (gt_scored,) = synthesizer.run_jobs(
            tokenized_map, tokenized_agent, [gt_job], neighbor_future=NEIGHBOR_FUTURE_LOGGED, **common
        )
    aux_log_prob = mean_token_log_prob(rollouts["token_log_prob"][:, 0], rollouts["next_token_valid"][:, 0])
    gt_log_prob = mean_token_log_prob(gt_scored["token_log_prob"][:, 0], gt_scored["next_token_valid"][:, 0])

    horizon = effective_horizon(data, cfg, rollouts["pred_traj_10hz"].shape[2])
    gt_ego = ground_truth_boxes(data, [obs.agent_idx], cfg.num_historical_steps, horizon)
    # the same source of the others' futures as at prediction time
    others = others_future_boxes(data, all_idx, cfg, horizon)
    origin = data["agent"]["position"][obs.agent_idx, cfg.num_historical_steps - 1, :2]

    aux_ego = rollout_boxes(rollouts, 0, data["agent"]["shape"][obs.agent_idx], horizon)
    # score sampled and logged trajectories over the same valid steps
    aux_ego.valid = gt_ego.valid.expand(len(aux_ego), -1)

    feature_kwargs = dict(
        map_points=map_points,
        n_circles=cfg.n_collision_circles,
        collision_margin=cfg.collision_margin,
        drivable_half_width=cfg.drivable_half_width,
    )
    real_feat = extract_trajectory_features(
        gt_ego, others, origin, log_prob=gt_log_prob, **feature_kwargs
    ).squeeze(0)
    fake_feat = extract_trajectory_features(
        aux_ego, others, origin, log_prob=aux_log_prob, **feature_kwargs
    )
    return real_feat, fake_feat

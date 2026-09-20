"""Builds a responsibility-augmented dataset (paper Sec. IV-B) from cached
WOMD scenarios: per-agent responsibility observations, HMM-based level
labels, and per-level (real vs. auxiliary) trajectory feature sets used to
train the reward networks in :mod:`src.responsibility.reward_model`.

Scoping note on "sequences". The paper computes a dense responsibility
*sequence* per agent by re-evaluating Eq. (3)/(4) at every 0.1s timestep
across nuScenes' 20s scenes, then fits the HMM on those long sequences.
This repo's cached WOMD scenarios are instead short, non-overlapping
~9.1s clips (one context per file), so by default we compute a single
responsibility observation per (scenario, queried agent) at the clip's
native current step (``num_historical_steps - 1``). Fed to
:class:`~src.responsibility.hmm.GaussianHMM`, length-1 sequences still
give a valid level *abstraction* (Sec. IV-A's emission model B), just
without a learned transition matrix A beyond the uniform prior. For
users who want genuine within-clip temporal sequences,
:func:`sliding_window_contexts` re-slices one cached scenario at several
shifted "current steps" k so the same agent yields a multi-step sequence;
this is an optional, less battle-tested extension (see its docstring).
"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from src.responsibility.hmm import GaussianHMM
from src.responsibility.metrics import (
    ResponsibilityConfig,
    compute_agent_responsibility,
    effective_horizon,
    ground_truth_boxes,
    rollout_boxes,
)
from src.responsibility.motion_set import MotionSetSynthesizer, find_neighbors
from src.responsibility.reward_model import extract_trajectory_features, trajectory_log_prob


@dataclass
class ResponsibilityObservation:
    scenario_id: str
    agent_id: int
    agent_idx: int
    k: int
    beta: Tensor  # [2]: (safety, courtesy)
    level: Optional[int] = None


def get_map_points(data) -> Optional[Tensor]:
    try:
        return data["map_save"]["traj_pos"][:, 0]  # [n_pl, 2], one sample point per segment
    except (KeyError, IndexError):
        return None


def compute_scenario_responsibility(
    synthesizer: MotionSetSynthesizer,
    data,
    cfg: ResponsibilityConfig,
    agent_type_filter: Optional[int] = 0,
    max_agents: Optional[int] = None,
) -> List[ResponsibilityObservation]:
    """One responsibility observation per queried agent at the scenario's
    native current step (Eq. 3/4/5), for every agent whose type matches
    ``agent_type_filter`` (default 0 = vehicle, matching the paper's
    vehicle-vehicle scope; pass ``None`` for all types)."""
    tokenized_map, tokenized_agent = synthesizer.tokenize(data)
    n_agent = data["agent"]["position"].shape[0]
    at_step = cfg.num_historical_steps - 1
    # data comes from a batch_size=1 DataLoader, so PyG collates
    # "scenario_id" (a per-graph, non-tensor attribute) into a length-1 list.
    # Callers that already know the scenario_id (e.g. compute_dataset_responsibility)
    # overwrite this afterward; this is only a fallback for standalone use.
    if "scenario_id" in data:
        raw_scenario_id = data["scenario_id"]
        scenario_id = str(raw_scenario_id[0]) if isinstance(raw_scenario_id, (list, tuple)) else str(raw_scenario_id)
    else:
        scenario_id = "unknown"
    map_points = get_map_points(data)

    candidates = [
        i
        for i in range(n_agent)
        if (agent_type_filter is None or int(data["agent"]["type"][i]) == agent_type_filter)
        and bool(data["agent"]["valid_mask"][i, at_step])
    ]
    if max_agents is not None:
        candidates = candidates[:max_agents]
    all_idx = list(range(n_agent))

    observations = []
    for a in candidates:
        neighbor_mask = find_neighbors(
            data["agent"]["position"][..., :2],
            data["agent"]["valid_mask"],
            a,
            at_step=at_step,
            radius=cfg.neighbor_radius,
        )
        neighbor_indices = neighbor_mask.nonzero(as_tuple=True)[0].tolist()
        result = compute_agent_responsibility(
            synthesizer,
            data,
            tokenized_map,
            tokenized_agent,
            a,
            neighbor_indices,
            all_idx,
            cfg,
            map_points=map_points,
        )
        observations.append(
            ResponsibilityObservation(
                scenario_id=str(scenario_id),
                agent_id=int(data["agent"]["id"][a]),
                agent_idx=a,
                k=at_step,
                beta=result["beta"].detach().cpu(),
            )
        )
    return observations


def sliding_window_contexts(
    data, k_values: Sequence[int], num_historical_steps: int = 11
):
    """Experimental: re-slices one cached scenario so index ``k`` becomes the
    new "current step" boundary, by shifting the agent time-series tensors.
    Returns a list of ``(k, shifted_data)``; only ``k >= num_historical_steps - 1``
    with enough trailing steps in the original clip are returned. Map/static
    fields are shared (not copied) with the original ``data``.
    """
    n_step = data["agent"]["position"].shape[1]
    time_fields = ["position", "heading", "velocity", "valid_mask"]
    results = []
    for k in k_values:
        start = k - (num_historical_steps - 1)
        if start < 0 or start >= n_step:
            continue
        shifted = copy.copy(data)
        shifted["agent"] = copy.copy(data["agent"])
        for f in time_fields:
            if f in data["agent"]:
                shifted["agent"][f] = data["agent"][f][:, start:]
        results.append((k, shifted))
    return results


def group_sequences(
    observations: Sequence[ResponsibilityObservation],
) -> Dict[Tuple[str, int], List[ResponsibilityObservation]]:
    groups: Dict[Tuple[str, int], List[ResponsibilityObservation]] = {}
    for obs in observations:
        key = (obs.scenario_id, obs.agent_id)
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
    buckets: Dict[int, Tuple[List[Tensor], List[Tensor]]] = {}

    for obs in observations:
        if obs.level is None:
            continue
        data = data_by_scenario[obs.scenario_id]
        tokenized_map, tokenized_agent = synthesizer.tokenize(data)
        n_agent = data["agent"]["position"].shape[0]
        all_idx = [i for i in range(n_agent) if i != obs.agent_idx]
        map_points = get_map_points(data)

        rollouts = synthesizer.sample_rollouts(
            tokenized_map,
            tokenized_agent,
            free_agent_indices=[obs.agent_idx],
            n_samples=n_aux,
            sampling_scheme=sampling_scheme,
            rollout_horizon=synthesizer.round_horizon(cfg.metric_horizon),
        )
        aux_log_prob = trajectory_log_prob(
            rollouts["next_token_logits"][:, 0],
            rollouts["next_token_idx"][:, 0],
            rollouts["next_token_valid"][:, 0],
        )  # [n_aux]
        # The logged trajectory's own model log-probability, from a pass in
        # which *every* agent replays ground truth (i.e. teacher forcing).
        # Leaving this at 0 for real trajectories and negative for sampled
        # ones would hand the discriminator a trivial shortcut feature.
        gt_scored = synthesizer.sample_rollouts(
            tokenized_map,
            tokenized_agent,
            free_agent_indices=[],
            n_samples=1,
            output_agent_indices=[obs.agent_idx],
            sampling_scheme=sampling_scheme,
            rollout_horizon=synthesizer.round_horizon(cfg.metric_horizon),
        )
        gt_log_prob = trajectory_log_prob(
            gt_scored["next_token_logits"][:, 0],
            gt_scored["next_token_idx"][:, 0],
            gt_scored["next_token_valid"][:, 0],
        )  # [1]

        horizon = effective_horizon(data, cfg, rollouts["pred_traj_10hz"].shape[2])
        gt_ego = ground_truth_boxes(data, [obs.agent_idx], cfg.num_historical_steps, horizon)
        others = ground_truth_boxes(data, all_idx, cfg.num_historical_steps, horizon)
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

        real_list, fake_list = buckets.setdefault(obs.level, ([], []))
        real_list.append(real_feat)
        fake_list.append(fake_feat)

    return {
        level: LevelFeatureSet(real=torch.stack(real_list), fake=torch.cat(fake_list, dim=0))
        for level, (real_list, fake_list) in buckets.items()
        if len(real_list) > 0
    }

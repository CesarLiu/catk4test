"""Online responsibility-aware trajectory prediction (paper Sec. IV-B, V-C):
given an agent's current responsibility level (Bayes-filtered from its
observed responsibility history, Eq. 6), select the sampled rollout that
maximizes that level's learned reward function.
"""

from typing import Dict, List, Optional

import torch
from torch import Tensor

from src.responsibility.dataset_builder import get_map_points
from src.responsibility.hmm import GaussianHMM
from src.responsibility.metrics import (
    ResponsibilityConfig,
    effective_horizon,
    ground_truth_boxes,
    rollout_boxes,
)
from src.responsibility.motion_set import MotionSetSynthesizer
from src.responsibility.reward_model import (
    RewardNetBundle,
    extract_trajectory_features,
    select_best_trajectory,
    trajectory_log_prob,
)


def current_responsibility_level(beta_history: Tensor, hmm: GaussianHMM) -> int:
    """z_k, Eq. (6): the most likely responsibility level given the
    observation history up to and including the current step."""
    posterior = hmm.forward_filter(beta_history.numpy())
    return int(posterior[-1].argmax())


def predict_responsibility_aware(
    synthesizer: MotionSetSynthesizer,
    data,
    query_idx: int,
    beta_history: Tensor,  # [T, 2], T >= 1
    hmm: GaussianHMM,
    reward_nets: Dict[int, RewardNetBundle],
    cfg: ResponsibilityConfig,
    n_candidates: int = 25,
    sampling_scheme=None,
    fallback_level: int = 0,
) -> Dict:
    """Samples ``n_candidates`` counterfactual/candidate rollouts for
    ``query_idx`` and returns the one maximizing the current level's reward
    network, argmax_xi R_{phi_h}(xi) (Sec. V-C)."""
    level = current_responsibility_level(beta_history, hmm)
    if level not in reward_nets:
        level = fallback_level if fallback_level in reward_nets else next(iter(reward_nets))

    tokenized_map, tokenized_agent = synthesizer.tokenize(data)
    n_agent = data["agent"]["position"].shape[0]
    all_idx = [i for i in range(n_agent) if i != query_idx]
    map_points = get_map_points(data)

    rollouts = synthesizer.sample_rollouts(
        tokenized_map,
        tokenized_agent,
        free_agent_indices=[query_idx],
        n_samples=n_candidates,
        sampling_scheme=sampling_scheme,
        # cfg.metric_horizon doubles as the prediction horizon: only those
        # steps are scored, and an autoregressive rollout truncated early is
        # exactly the prefix of the full-length one.
        rollout_horizon=synthesizer.round_horizon(cfg.metric_horizon),
    )
    traj = rollouts["pred_traj_10hz"][:, 0]  # [n_candidates, T, 2]
    log_prob = trajectory_log_prob(
        rollouts["next_token_logits"][:, 0],
        rollouts["next_token_idx"][:, 0],
        rollouts["next_token_valid"][:, 0],
    )

    horizon = effective_horizon(data, cfg, traj.shape[1])
    origin = data["agent"]["position"][query_idx, cfg.num_historical_steps - 1, :2]
    gt_ego = ground_truth_boxes(data, [query_idx], cfg.num_historical_steps, horizon)
    others = ground_truth_boxes(data, all_idx, cfg.num_historical_steps, horizon)

    ego = rollout_boxes(rollouts, 0, data["agent"]["shape"][query_idx], horizon)
    ego.valid = gt_ego.valid.expand(n_candidates, -1)

    features = extract_trajectory_features(
        ego,
        others,
        origin,
        map_points,
        log_prob=log_prob,
        n_circles=cfg.n_collision_circles,
        collision_margin=cfg.collision_margin,
        drivable_half_width=cfg.drivable_half_width,
    )

    bundle = reward_nets[level]
    best_idx = select_best_trajectory(bundle, features)

    return {
        "level": level,
        "best_idx": best_idx,
        "horizon": horizon,
        "selected_traj": traj[best_idx],
        "candidates": traj,
        "features": features,
        "rewards": bundle.reward(features).detach(),
    }

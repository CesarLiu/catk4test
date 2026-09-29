"""Online responsibility-aware trajectory prediction (paper Sec. IV-B, V-C):
given an agent's current responsibility level (Bayes-filtered from its
observed responsibility history, Eq. 6), select the sampled rollout that
maximizes that level's learned reward function.
"""

from dataclasses import dataclass
from typing import Dict

from torch import Tensor

from src.responsibility.dataset_builder import get_map_points
from src.responsibility.hmm import GaussianHMM
from src.responsibility.metrics import (
    ResponsibilityConfig,
    others_future_boxes,
    rollout_boxes,
)
from src.responsibility.motion_set import MotionSetSynthesizer, RolloutJob
from src.responsibility.reward_model import (
    RewardNetBundle,
    extract_trajectory_features,
    mean_token_log_prob,
    select_best_trajectory,
)


def current_responsibility_level(beta_history: Tensor, hmm: GaussianHMM) -> int:
    """z_k, Eq. (6): the most likely responsibility level given the
    observation history up to and including the current step."""
    posterior = hmm.forward_filter(beta_history.detach().cpu().numpy())
    return int(posterior[-1].argmax())


@dataclass
class Candidates:
    """Candidate futures of one agent at the current step, with the
    features the reward networks score them by."""

    traj: Tensor  # [K, T, 2]
    features: Tensor  # [K, 4]: collision, road deviation, progress, log-prob
    log_prob: Tensor  # [K], mean per-step model log-probability
    horizon: int


def sample_candidates(
    synthesizer: MotionSetSynthesizer,
    data,
    query_idx: int,
    cfg: ResponsibilityConfig,
    n_candidates: int = 25,
    sampling_scheme=None,
) -> Candidates:
    """Draws ``n_candidates`` futures of ``query_idx`` from the model and
    featurises them. Only information available now enters the features:
    the other agents' futures come from ``cfg.others_future``, which should
    not be "logged" when this is used to predict."""
    tokenized_map, tokenized_agent = synthesizer.tokenize(data)
    n_agent = data["agent"]["position"].shape[0]
    others_idx = [i for i in range(n_agent) if i != query_idx]

    (rollouts,) = synthesizer.run_jobs(
        tokenized_map,
        tokenized_agent,
        [RolloutJob(free=(query_idx,), n_samples=n_candidates)],
        sampling_scheme=sampling_scheme,
        # cfg.metric_horizon doubles as the prediction horizon: only those
        # steps are scored, and an autoregressive rollout truncated early is
        # exactly the prefix of the full-length one.
        rollout_horizon=synthesizer.round_horizon(cfg.metric_horizon),
        neighbor_future=cfg.neighbor_future,
    )
    traj = rollouts["pred_traj_10hz"][:, 0]  # [K, T, 2]
    log_prob = mean_token_log_prob(rollouts["token_log_prob"][:, 0], rollouts["next_token_valid"][:, 0])

    horizon = min(cfg.metric_horizon, traj.shape[1])
    ego = rollout_boxes(rollouts, 0, data["agent"]["shape"][query_idx], horizon)
    features = extract_trajectory_features(
        ego,
        others_future_boxes(data, others_idx, cfg, horizon),
        data["agent"]["position"][query_idx, cfg.num_historical_steps - 1, :2],
        get_map_points(data),
        log_prob=log_prob,
        n_circles=cfg.n_collision_circles,
        collision_margin=cfg.collision_margin,
        drivable_half_width=cfg.drivable_half_width,
    )
    return Candidates(traj=traj[:, :horizon], features=features, log_prob=log_prob, horizon=horizon)


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
    """Samples ``n_candidates`` candidate futures for ``query_idx`` and
    returns the one maximizing the current level's reward network,
    argmax_xi R_{phi_h}(xi) (Sec. V-C)."""
    level = current_responsibility_level(beta_history, hmm)
    if level not in reward_nets:
        level = fallback_level if fallback_level in reward_nets else next(iter(reward_nets))

    cand = sample_candidates(synthesizer, data, query_idx, cfg, n_candidates, sampling_scheme)
    bundle = reward_nets[level]
    best_idx = select_best_trajectory(bundle, cand.features)
    return {
        "level": level,
        "best_idx": best_idx,
        "horizon": cand.horizon,
        "selected_traj": cand.traj[best_idx],
        "candidates": cand.traj,
        "features": cand.features,
        "rewards": bundle.reward(cand.features).detach(),
    }

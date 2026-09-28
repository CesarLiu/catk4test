"""Responsibility-dependent reward functions (paper Sec. IV-B / V-C).

For each discrete responsibility level h, we train a small reward network
R_{phi_h} to discriminate real logged trajectories from auxiliary
(model-sampled) ones via the binary cross-entropy IRL objective in the
paper. The four trajectory features (collision cost, road-deviation cost,
final-displacement/progress, and the model's own predicted log-probability
of the trajectory) mirror the paper's Sec. V-C feature set.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.responsibility.geometry import (
    BoxTrajectories,
    collision_cost,
    final_displacement_progress,
    road_deviation_cost,
)


class RewardMLP(nn.Module):
    """Two-layer, 32-unit MLP reward network, matching Sec. V-C."""

    def __init__(self, in_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:  # [..., in_dim] -> [...] logits
        return self.net(features).squeeze(-1)

    def reward(self, features: Tensor) -> Tensor:  # [..., in_dim] -> [...] in (0, 1)
        return torch.sigmoid(self.forward(features))


@dataclass
class RewardNetBundle:
    model: RewardMLP
    feature_mean: Tensor
    feature_std: Tensor

    def reward(self, raw_features: Tensor) -> Tensor:
        # the nets are tiny and live on CPU; features may come from a GPU rollout
        device = raw_features.device
        normed = (raw_features.to(self.feature_mean.device) - self.feature_mean) / self.feature_std
        return self.model.reward(normed).to(device)

    def state_dict(self) -> Dict:
        return {
            "model": self.model.state_dict(),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "in_dim": self.feature_mean.shape[-1],
        }

    @classmethod
    def load_state_dict(cls, state: Dict, hidden_dim: int = 32) -> "RewardNetBundle":
        model = RewardMLP(in_dim=state["in_dim"], hidden_dim=hidden_dim)
        model.load_state_dict(state["model"])
        return cls(model=model, feature_mean=state["feature_mean"], feature_std=state["feature_std"])


def trajectory_log_prob(
    next_token_logits: Tensor,  # [..., 16, n_token]
    next_token_idx: Tensor,  # [..., 16]
    next_token_valid: Tensor,  # [..., 16]
) -> Tensor:  # [...]
    """Mean per-step log pi_theta(realized token | context) over valid
    future steps -- the "predicted probability" feature of Sec. V-C,
    evaluated on the token actually realized by the rollout rather than
    the step-wise arg-max (which need not be the token that was sampled).
    """
    log_probs = torch.log_softmax(next_token_logits, dim=-1)
    gathered = torch.gather(log_probs, -1, next_token_idx.unsqueeze(-1)).squeeze(-1)
    valid = next_token_valid.to(gathered.dtype)
    return (gathered * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1.0)


def extract_trajectory_features(
    ego: BoxTrajectories,
    others: BoxTrajectories,
    origin: Tensor,  # [2]
    map_points: Optional[Tensor],
    log_prob: Optional[Tensor] = None,  # [N]
    n_circles: int = 3,
    collision_margin: float = 0.0,
    drivable_half_width: float = 3.0,
) -> Tensor:  # [N, 4]: (collision_cost, road_dev_cost, progress, log_prob)
    c_coll = collision_cost(ego, others, n_circles=n_circles, margin=collision_margin)

    if map_points is not None and map_points.numel() > 0:
        c_dev = road_deviation_cost(ego.pos, ego.valid, map_points, drivable_half_width)
    else:
        c_dev = torch.zeros_like(c_coll)

    progress = final_displacement_progress(ego.pos, ego.valid, origin)

    if log_prob is None:
        log_prob = torch.zeros_like(c_coll)
    return torch.stack([c_coll, c_dev, progress, log_prob], dim=-1)


def train_reward_network(
    features_real: Tensor,  # [N, F]
    features_fake: Tensor,  # [M, F]
    hidden_dim: int = 32,
    n_epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> Tuple[RewardNetBundle, float]:
    """Trains one level's reward network via the BCE discriminator loss:
    L(phi) = -sum y*log(R(xi)) + (1-y)*log(1-R(xi)), y=1 for real trajectories.
    """
    torch.manual_seed(seed)
    features_real, features_fake = features_real.detach().cpu(), features_fake.detach().cpu()
    all_features = torch.cat([features_real, features_fake], dim=0)
    feature_mean = all_features.mean(dim=0)
    feature_std = all_features.std(dim=0).clamp(min=1e-6)

    norm_real = (features_real - feature_mean) / feature_std
    norm_fake = (features_fake - feature_mean) / feature_std
    inputs = torch.cat([norm_real, norm_fake], dim=0)
    labels = torch.cat(
        [torch.ones(len(norm_real)), torch.zeros(len(norm_fake))]
    )

    model = RewardMLP(in_dim=features_real.shape[-1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    final_loss = torch.tensor(float("nan"))
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        final_loss = loss.detach()

    bundle = RewardNetBundle(model=model, feature_mean=feature_mean, feature_std=feature_std)
    return bundle, float(final_loss)


def train_reward_networks_per_level(
    features_by_level: Dict[int, Tuple[Tensor, Tensor]],
    hidden_dim: int = 32,
    n_epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> Dict[int, RewardNetBundle]:
    """features_by_level[h] = (features_real, features_fake) for level h's cluster D_h."""
    bundles = {}
    for level, (real, fake) in features_by_level.items():
        if real.shape[0] == 0 or fake.shape[0] == 0:
            continue
        bundle, _ = train_reward_network(
            real,
            fake,
            hidden_dim=hidden_dim,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
        )
        bundles[level] = bundle
    return bundles


def select_best_trajectory(bundle: RewardNetBundle, candidate_features: Tensor) -> int:
    """argmax_xi R_{phi_h}(xi), used for online responsibility-aware selection (Sec. V-C)."""
    rewards = bundle.reward(candidate_features)
    return int(torch.argmax(rewards).item())

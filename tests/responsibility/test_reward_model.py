import torch

from src.responsibility.geometry import BoxTrajectories, collision_cost
from src.responsibility.reward_model import (
    RewardMLP,
    extract_trajectory_features,
    select_best_trajectory,
    train_reward_network,
    trajectory_log_prob,
)


def _boxes(pos, heading=None, valid=None, length=4.8, width=2.0):
    n, t = pos.shape[0], pos.shape[1]
    return BoxTrajectories(
        pos=pos,
        heading=torch.zeros(n, t) if heading is None else heading,
        valid=torch.ones(n, t, dtype=torch.bool) if valid is None else valid,
        length=torch.full((n,), length),
        width=torch.full((n,), width),
    )


def test_reward_mlp_output_shapes():
    model = RewardMLP(in_dim=4, hidden_dim=8)
    features = torch.randn(5, 4)
    logits = model(features)
    reward = model.reward(features)
    assert logits.shape == (5,)
    assert reward.shape == (5,)
    assert (reward >= 0).all() and (reward <= 1).all()


def test_train_reward_network_separates_clusters():
    torch.manual_seed(0)
    real = torch.randn(64, 4) * 0.1 + torch.tensor([0.0, 0.0, 5.0, 0.0])  # low cost, high progress
    fake = torch.randn(64, 4) * 0.1 + torch.tensor([5.0, 5.0, 0.0, -5.0])  # high cost, low progress

    bundle, final_loss = train_reward_network(real, fake, hidden_dim=16, n_epochs=200, lr=1e-2)
    assert final_loss < 0.5

    reward_real = bundle.reward(real)
    reward_fake = bundle.reward(fake)
    assert reward_real.mean() > reward_fake.mean()


def test_select_best_trajectory_picks_highest_reward():
    torch.manual_seed(1)
    real = torch.randn(32, 4) * 0.05 + torch.tensor([0.0, 0.0, 5.0, 0.0])
    fake = torch.randn(32, 4) * 0.05 + torch.tensor([5.0, 5.0, 0.0, -5.0])
    bundle, _ = train_reward_network(real, fake, hidden_dim=16, n_epochs=200, lr=1e-2)

    candidates = torch.stack(
        [torch.tensor([5.0, 5.0, 0.0, -5.0]), torch.tensor([0.0, 0.0, 5.0, 0.0])]
    )
    best_idx = select_best_trajectory(bundle, candidates)
    assert best_idx == 1


def test_extract_trajectory_features_shape_and_values():
    T = 4
    ego = _boxes(torch.zeros(3, T, 2))  # 3 candidate trajectories at the origin
    others = _boxes(torch.full((2, T, 2), 100.0))  # far away -> no collision

    features = extract_trajectory_features(ego, others, torch.zeros(2), map_points=None)
    assert features.shape == (3, 4)
    assert torch.allclose(features[:, 0], torch.zeros(3))  # no collision
    assert torch.allclose(features[:, 2], torch.zeros(3))  # no displacement from origin


def test_collision_cost_used_consistently_by_features():
    T = 2
    ego = _boxes(torch.zeros(1, T, 2))
    others = _boxes(torch.zeros(1, T, 2))
    cost = collision_cost(ego, others)
    features = extract_trajectory_features(ego, others, torch.zeros(2), map_points=None)
    assert torch.allclose(features[:, 0], cost)


def test_trajectory_log_prob_matches_manual_gather():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 6)
    idx = torch.tensor([[0, 1, 2, 3], [5, 4, 3, 2]])
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])

    out = trajectory_log_prob(logits, idx, valid)

    log_probs = torch.log_softmax(logits, dim=-1)
    expected_0 = sum(log_probs[0, t, idx[0, t]] for t in range(4)) / 4
    expected_1 = sum(log_probs[1, t, idx[1, t]] for t in range(2)) / 2
    assert torch.isclose(out[0], expected_0)
    assert torch.isclose(out[1], expected_1)


def test_trajectory_log_prob_is_higher_for_the_likely_token():
    logits = torch.zeros(1, 1, 3)
    logits[0, 0, 0] = 5.0
    valid = torch.ones(1, 1, dtype=torch.bool)
    likely = trajectory_log_prob(logits, torch.tensor([[0]]), valid)
    unlikely = trajectory_log_prob(logits, torch.tensor([[1]]), valid)
    assert likely.item() > unlikely.item()

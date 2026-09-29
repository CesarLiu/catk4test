import pytest
import torch

from src.responsibility.metrics import ResponsibilityConfig, plausible_sample_indices, safety_from_rollouts


def test_plausible_indices_keep_the_most_probable_in_order():
    lp = torch.tensor([[-1.0, -1.0], [-9.0, -9.0], [-2.0, -2.0], [-0.5, -0.5]])
    valid = torch.ones_like(lp, dtype=torch.bool)
    assert plausible_sample_indices(lp, valid, 0.5).tolist() == [0, 3]
    assert plausible_sample_indices(lp, valid, 1.0).tolist() == [0, 1, 2, 3]
    assert plausible_sample_indices(lp, valid, 0.01).tolist() == [3]  # at least one


def test_plausible_indices_average_over_valid_tokens_only():
    lp = torch.tensor([[-1.0, -50.0], [-2.0, -2.0]])
    valid = torch.tensor([[True, False], [True, True]])  # sample 0's bad token is padding
    assert plausible_sample_indices(lp, valid, 0.5).tolist() == [0]


def test_plausible_fraction_is_validated():
    lp = torch.zeros(3, 2)
    with pytest.raises(ValueError):
        plausible_sample_indices(lp, torch.ones_like(lp, dtype=torch.bool), 0.0)


def _straight_scene(n_hist=11, horizon=20):
    """Agent 0 drives along y=0, agent 1 along y=6 (both logged)."""
    t = n_hist + horizon
    x = torch.arange(t, dtype=torch.float32)
    pos = torch.zeros(2, t, 3)
    pos[:, :, 0] = x
    pos[1, :, 1] = 6.0
    return {"agent": {
        "position": pos,
        "heading": torch.zeros(2, t),
        "valid_mask": torch.ones(2, t, dtype=torch.bool),
        "shape": torch.tensor([[4.5, 2.0, 1.5], [4.5, 2.0, 1.5]]),
    }}


def test_implausible_outlier_no_longer_drives_the_safety_cvar():
    n_hist, horizon = 11, 20
    data = _straight_scene(n_hist, horizon)
    gt = data["agent"]["position"][0, n_hist:, :2]
    traj = gt.expand(10, -1, -1).clone()
    traj[0, :, 1] = -10.0  # one erratic sample veers far away from the neighbour ...
    lp = torch.full((10, 1, 4), -1.0)
    lp[0] = -12.0  # ... and the model finds it very unlikely
    rollouts = {
        "pred_traj_10hz": traj[:, None],
        "token_log_prob": lp,
        "next_token_valid": torch.ones(10, 1, 4, dtype=torch.bool),
    }
    cfg = ResponsibilityConfig(num_historical_steps=n_hist, metric_horizon=horizon, cvar_alpha=0.1, d_sat=None)
    everything = safety_from_rollouts(rollouts, data, 0, [1], cfg)["value"]
    plausible = safety_from_rollouts(
        rollouts, data, 0, [1], ResponsibilityConfig(**{**cfg.__dict__, "plausible_fraction": 0.9})
    )["value"]
    # ... which makes the logged motion look like it gave up margin it need not have
    assert float(everything) > 0.0
    assert float(plausible) == pytest.approx(0.0, abs=1e-5)

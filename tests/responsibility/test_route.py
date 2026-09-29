import math

import pytest
import torch

from src.responsibility.geometry import lateral_deviation, logged_route
from src.responsibility.metrics import (
    ResponsibilityConfig,
    route_consistent_sample_indices,
    safety_from_rollouts,
)
from tests.responsibility.test_plausible import _straight_scene

N_HIST, HORIZON = 11, 20


def test_lateral_deviation_measures_distance_off_the_path_not_along_it():
    path = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])  # straight, then a left turn
    pts = torch.tensor([[5.0, 0.0], [5.0, 2.0], [12.0, 5.0], [-3.0, 4.0]])
    assert torch.allclose(lateral_deviation(pts, path), torch.tensor([0.0, 2.0, 2.0, 5.0]))


def test_logged_route_extends_along_the_final_heading():
    future = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
    valid = torch.tensor([True, True, False])
    route = logged_route(torch.zeros(2), future, valid, torch.tensor(math.pi / 2), extension=10.0)
    assert torch.allclose(route, torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 10.0]]), atol=1e-6)
    # an agent with no logged future still gets a path straight ahead
    stopped = logged_route(torch.zeros(2), future, torch.zeros(3, dtype=torch.bool), torch.tensor(0.0), 10.0)
    assert torch.allclose(stopped, torch.tensor([[0.0, 0.0], [10.0, 0.0]]))


def _samples(scene):
    """Agent 0 drives +x at 1 m per step. Three alternatives: brake, speed
    up past the end of the log, and turn off to the right."""
    gt = scene["agent"]["position"][0, N_HIST:, :2]
    t = torch.arange(HORIZON).float()  # all start where the log does
    brake = gt.clone()
    brake[:, 0] = gt[0, 0] + 0.3 * t
    faster = gt.clone()
    faster[:, 0] = gt[0, 0] + 2.0 * t
    turn = gt.clone()
    turn[:, 1] = -1.0 - 0.5 * t  # over 10 m off the route by the end
    return torch.stack([brake, faster, turn])


def test_speed_is_free_but_the_route_is_fixed():
    scene = _straight_scene(N_HIST, HORIZON)
    cfg = ResponsibilityConfig(num_historical_steps=N_HIST, metric_horizon=HORIZON, route_tolerance=1.75)
    keep = route_consistent_sample_indices(_samples(scene), scene, 0, cfg, HORIZON)
    assert keep.tolist() == [0, 1]


def test_a_stopped_agent_keeps_straight_ahead():
    scene = _straight_scene(N_HIST, HORIZON)
    scene["agent"]["position"][0, N_HIST - 1 :, 0] = 0.0  # logged as standing still at the origin
    cfg = ResponsibilityConfig(num_historical_steps=N_HIST, metric_horizon=HORIZON, route_tolerance=1.75)
    samples = torch.zeros(2, HORIZON, 2)
    samples[0, :, 0] = torch.arange(HORIZON).float()  # pulls away straight ahead
    samples[1, :, 1] = torch.arange(HORIZON).float()  # turns left
    assert route_consistent_sample_indices(samples, scene, 0, cfg, HORIZON).tolist() == [0]


def test_when_nothing_follows_the_route_the_closest_sample_is_kept():
    scene = _straight_scene(N_HIST, HORIZON)
    cfg = ResponsibilityConfig(num_historical_steps=N_HIST, metric_horizon=HORIZON, route_tolerance=0.1)
    samples = _samples(scene)[[2, 2]]
    samples[1, :, 1] *= 0.5
    assert route_consistent_sample_indices(samples, scene, 0, cfg, HORIZON).tolist() == [1]


def test_turning_away_no_longer_counts_as_an_alternative():
    """Neighbour on the left (y=6); the alternative that gains the most
    margin is turning off to the right, which the route filter removes."""
    scene = _straight_scene(N_HIST, HORIZON)
    samples = torch.cat([_samples(scene)[[0, 1]], _samples(scene)[[2]].expand(8, -1, -1)])
    n = samples.shape[0]
    rollouts = {
        "pred_traj_10hz": samples[:, None],
        "token_log_prob": torch.full((n, 1, 4), -1.0),
        "next_token_valid": torch.ones(n, 1, 4, dtype=torch.bool),
    }
    base = dict(num_historical_steps=N_HIST, metric_horizon=HORIZON, cvar_alpha=0.1, d_sat=None)
    free = safety_from_rollouts(rollouts, scene, 0, [1], ResponsibilityConfig(**base))["value"]
    routed = safety_from_rollouts(
        rollouts, scene, 0, [1], ResponsibilityConfig(**base, route_tolerance=1.75)
    )["value"]
    # braking / speeding up along the lane barely change the gap to the
    # neighbour alongside; turning away widens it
    assert float(routed) < 0.1
    assert float(free) > float(routed) + 0.3


def test_route_filter_runs_before_the_plausibility_filter():
    scene = _straight_scene(N_HIST, HORIZON)
    samples = _samples(scene)  # brake, faster, turn
    lp = torch.tensor([-1.0, -5.0, -0.1])[:, None, None].expand(3, 1, 4).clone()
    rollouts = {
        "pred_traj_10hz": samples[:, None],
        "token_log_prob": lp,
        "next_token_valid": torch.ones(3, 1, 4, dtype=torch.bool),
    }
    cfg = ResponsibilityConfig(num_historical_steps=N_HIST, metric_horizon=HORIZON, cvar_alpha=0.1,
                               d_sat=None, route_tolerance=1.75, plausible_fraction=0.5)
    # the turn is the most probable sample but leaves the route; of the two
    # route-following ones, the more plausible half is braking alone
    value = safety_from_rollouts(rollouts, scene, 0, [1], cfg)["value"]
    brake_only = {k: v[:1] for k, v in rollouts.items()}
    plain = ResponsibilityConfig(num_historical_steps=N_HIST, metric_horizon=HORIZON, cvar_alpha=0.1, d_sat=None)
    assert float(value) == pytest.approx(float(safety_from_rollouts(brake_only, scene, 0, [1], plain)["value"]))

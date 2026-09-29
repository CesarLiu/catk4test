import math

import torch

from src.responsibility.geometry import (
    BoxTrajectories,
    circle_decomposition,
    collision_cost,
    final_displacement_progress,
    pairwise_min_distance_over_time,
    road_deviation_cost,
    safety_margin,
)

CAR_LENGTH, CAR_WIDTH = 4.8, 2.0


def _boxes(pos, heading=None, valid=None, length=CAR_LENGTH, width=CAR_WIDTH):
    n, t = pos.shape[0], pos.shape[1]
    return BoxTrajectories(
        pos=pos,
        heading=torch.zeros(n, t) if heading is None else heading,
        valid=torch.ones(n, t, dtype=torch.bool) if valid is None else valid,
        length=torch.full((n,), length),
        width=torch.full((n,), width),
    )


def _straight(x, y, n_step=3):
    """One trajectory sitting still at (x, y), heading +x."""
    pos = torch.zeros(1, n_step, 2)
    pos[..., 0] = x
    pos[..., 1] = y
    return pos


def test_safety_margin_basic():
    T = 5
    ref = torch.zeros(T, 2)
    traj = torch.stack([torch.arange(T).float(), torch.zeros(T)], dim=-1)
    assert torch.isclose(safety_margin(traj, ref), torch.tensor(0.0))


def test_safety_margin_batched_and_masking():
    T = 4
    ref = torch.zeros(T, 2)
    traj = torch.zeros(2, T, 2)
    traj[0] = torch.tensor([[5.0, 0], [5.0, 0], [5.0, 0], [5.0, 0]])
    traj[1] = torch.tensor([[5.0, 0], [5.0, 0], [0.1, 0], [5.0, 0]])
    out = safety_margin(traj, ref)
    assert out.shape == (2,)
    assert torch.isclose(out[0], torch.tensor(5.0))
    assert torch.isclose(out[1], torch.tensor(0.1))

    valid = torch.tensor([True, True, False, True])
    out_masked = safety_margin(traj, ref, ref_valid=valid)
    assert torch.isclose(out_masked[1], torch.tensor(5.0))


def test_safety_margin_saturation():
    T = 3
    ref = torch.zeros(T, 2)
    far = torch.full((1, T, 2), 40.0)
    assert torch.isclose(safety_margin(far, ref), torch.linalg.norm(torch.tensor([40.0, 40.0])))
    assert torch.isclose(safety_margin(far, ref, d_sat=10.0), torch.tensor(10.0))

    near = torch.full((1, T, 2), 1.0)
    unsaturated = safety_margin(near, ref)
    assert torch.isclose(safety_margin(near, ref, d_sat=10.0), unsaturated)


def test_saturation_bounds_the_safety_decrease():
    """With saturation, a far-away neighbour cannot inflate the metric:
    all alternatives and the real trajectory alike cap out at d_sat."""
    T = 3
    ref = torch.zeros(T, 2)
    real = torch.full((1, T, 2), 30.0)
    alternative = torch.full((1, T, 2), 60.0)
    decrease = safety_margin(alternative, ref, d_sat=10.0) - safety_margin(real, ref, d_sat=10.0)
    assert torch.isclose(decrease, torch.tensor(0.0))


def test_pairwise_min_distance_matches_safety_margin():
    T = 3
    a = torch.rand(4, T, 2)
    b = torch.rand(T, 2)
    assert torch.allclose(pairwise_min_distance_over_time(a, b), safety_margin(a, b))


def test_circle_decomposition_covers_the_box():
    length = torch.tensor([CAR_LENGTH])
    width = torch.tensor([CAR_WIDTH])
    offsets, radius = circle_decomposition(length, width, n_circles=3)

    assert torch.allclose(offsets, torch.tensor([[-1.6, 0.0, 1.6]]), atol=1e-5)
    assert torch.isclose(radius, torch.tensor(math.sqrt(0.8**2 + 1.0**2)), atol=1e-5)
    # every box corner must be inside some circle
    corners = torch.tensor([[2.4, 1.0], [2.4, -1.0], [-2.4, 1.0], [-2.4, -1.0]])
    for corner in corners:
        d = torch.abs(corner[0] - offsets[0]) ** 2 + corner[1] ** 2
        assert torch.sqrt(d).min() <= radius[0] + 1e-5


def test_no_collision_between_adjacent_lanes():
    """Regression: a single circle of radius length/2 made two cars in
    adjacent lanes permanently 'collide', polluting the courtesy KL and
    the IRL collision feature."""
    ego = _boxes(_straight(0.0, 0.0))
    other = _boxes(_straight(0.0, 3.5))
    assert torch.isclose(collision_cost(ego, other), torch.tensor(0.0))


def test_collision_on_overlap():
    ego = _boxes(_straight(0.0, 0.0))
    other = _boxes(_straight(0.0, 0.0))
    assert collision_cost(ego, other).item() > 0.0


def test_rear_end_gap_sensitivity():
    ego = _boxes(_straight(0.0, 0.0))
    close = _boxes(_straight(5.0, 0.0))  # 0.2 m bumper gap
    far = _boxes(_straight(8.0, 0.0))  # 3.2 m bumper gap
    assert collision_cost(ego, close).item() > 0.0
    assert torch.isclose(collision_cost(ego, far), torch.tensor(0.0))


def test_collision_respects_validity():
    ego = _boxes(_straight(0.0, 0.0))
    other = _boxes(_straight(0.0, 0.0))
    other.valid[:] = False
    assert torch.isclose(collision_cost(ego, other), torch.tensor(0.0))


def test_collision_batches_over_ego_trajectories():
    pos = torch.cat([_straight(0.0, 0.0), _straight(0.0, 3.5)], dim=0)
    ego = _boxes(pos)
    other = _boxes(_straight(0.0, 0.0))
    cost = collision_cost(ego, other)
    assert cost.shape == (2,)
    assert cost[0].item() > 0.0
    assert torch.isclose(cost[1], torch.tensor(0.0))


def test_collision_with_no_others_is_zero():
    ego = _boxes(_straight(0.0, 0.0))
    empty = BoxTrajectories(
        pos=torch.zeros(0, 3, 2),
        heading=torch.zeros(0, 3),
        valid=torch.zeros(0, 3, dtype=torch.bool),
        length=torch.zeros(0),
        width=torch.zeros(0),
    )
    assert torch.isclose(collision_cost(ego, empty), torch.tensor(0.0))


def test_heading_rotates_the_footprint():
    """Two cars side by side laterally do not collide; the same pair with
    one rotated 90 degrees (so it spans the lanes) does."""
    ego = _boxes(_straight(0.0, 0.0))
    other_pos = _straight(0.0, 3.0)
    parallel = _boxes(other_pos)
    perpendicular = _boxes(other_pos, heading=torch.full((1, 3), math.pi / 2))
    assert torch.isclose(collision_cost(ego, parallel), torch.tensor(0.0))
    assert collision_cost(ego, perpendicular).item() > 0.0


def test_box_truncate():
    ego = _boxes(_straight(0.0, 0.0, n_step=10))
    short = ego.truncate(4)
    assert short.horizon == 4
    assert ego.truncate(20) is ego


def test_road_deviation_cost_zero_when_near_map_points():
    T = 3
    traj = torch.zeros(1, T, 2)
    valid = torch.ones(1, T, dtype=torch.bool)
    map_points = torch.zeros(10, 2)
    cost = road_deviation_cost(traj, valid, map_points, drivable_half_width=3.0)
    assert torch.isclose(cost, torch.tensor([0.0]))


def test_road_deviation_cost_positive_far_from_map():
    T = 2
    traj = torch.full((1, T, 2), 1000.0)
    valid = torch.ones(1, T, dtype=torch.bool)
    map_points = torch.zeros(4, 2)
    cost = road_deviation_cost(traj, valid, map_points, drivable_half_width=3.0)
    assert cost.item() > 0.0


def test_final_displacement_progress():
    T = 4
    traj = torch.stack([torch.arange(T).float(), torch.zeros(T)], dim=-1).unsqueeze(0)
    valid = torch.ones(1, T, dtype=torch.bool)
    progress = final_displacement_progress(traj, valid, torch.zeros(2))
    assert torch.isclose(progress, torch.tensor([3.0]))


def test_final_displacement_progress_respects_validity():
    T = 4
    traj = torch.stack([torch.arange(T).float(), torch.zeros(T)], dim=-1).unsqueeze(0)
    valid = torch.tensor([[True, True, False, False]])
    progress = final_displacement_progress(traj, valid, torch.zeros(2))
    assert torch.isclose(progress, torch.tensor([1.0]))


def test_map_points_keep_only_vehicle_lane_centerlines():
    import tests.responsibility.conftest  # noqa: F401  (stub before importing dataset_builder)
    from src.responsibility.dataset_builder import get_map_points

    traj_pos = torch.arange(5 * 3 * 2, dtype=torch.float).view(5, 3, 2)
    data = {
        "map_save": {"traj_pos": traj_pos},
        # polygons: lane(freeway), lane(bike), road_edge, lane(surface street), crosswalk
        "pt_token": {"pl_type": torch.tensor([0, 0, 1, 0, 3]), "type": torch.tensor([0, 3, 4, 1, 9])},
    }
    points = get_map_points(data)
    assert points.shape == (2 * 3, 2)  # polylines 0 and 3, all three samples each
    assert torch.equal(points, traj_pos[[0, 3]].reshape(-1, 2))
    assert get_map_points(data, vehicle_lanes_only=False).shape == (5 * 3, 2)


def test_map_points_fall_back_when_no_vehicle_lane_exists():
    import tests.responsibility.conftest  # noqa: F401
    from src.responsibility.dataset_builder import get_map_points

    traj_pos = torch.zeros(2, 3, 2)
    data = {"map_save": {"traj_pos": traj_pos},
            "pt_token": {"pl_type": torch.tensor([3, 1]), "type": torch.tensor([9, 4])}}
    assert get_map_points(data).shape == (6, 2)

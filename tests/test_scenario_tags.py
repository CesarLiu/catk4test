import math

import pytest
import torch

from src.responsibility.scenario_tags import (
    PAIRS_ALL,
    PAIRS_AUTO,
    PAIRS_INTEREST,
    PAIRS_SDC,
    TagConfig,
    agent_maneuver,
    classify_interaction,
    classify_scene,
    interacting_pairs,
    interaction_window,
    tag_pair,
)

N_STEP = 91
DT = 0.1


def _follow_path(waypoints, speed, start_time=0.0):
    """Positions/headings at 10Hz of an agent driving the polyline
    ``waypoints`` at constant ``speed`` from ``start_time`` (it waits at the
    first waypoint before that, and stops at the last one)."""
    pts = torch.tensor(waypoints, dtype=torch.float)
    dense = [pts[0]]
    for p, q in zip(pts[:-1], pts[1:]):
        n = max(int(torch.linalg.norm(q - p) / 0.05), 1)
        for k in range(1, n + 1):
            dense.append(p + (q - p) * k / n)
    dense = torch.stack(dense)
    seg = torch.linalg.norm(dense[1:] - dense[:-1], dim=-1)
    arc = torch.cat([torch.zeros(1), seg.cumsum(0)])
    t = torch.arange(N_STEP).float() * DT
    s = ((t - start_time).clamp(min=0) * speed).clamp(max=float(arc[-1]))
    idx = torch.searchsorted(arc, s).clamp(max=len(dense) - 1)
    pos = dense[idx]
    d = dense[(idx + 1).clamp(max=len(dense) - 1)] - dense[(idx - 1).clamp(min=0)]
    head = torch.atan2(d[:, 1], d[:, 0])
    vel = torch.zeros(N_STEP, 2)
    vel[1:] = (pos[1:] - pos[:-1]) / DT
    vel[0] = vel[1]
    return pos, head, vel


def _arc(center, radius, a0, a1, n=12):
    return [[center[0] + radius * math.cos(a0 + (a1 - a0) * k / n),
             center[1] + radius * math.sin(a0 + (a1 - a0) * k / n)] for k in range(n + 1)]


def _scene(tracks, types=None, roles=None, map_save=None):
    n = len(tracks)
    pos = torch.zeros(n, N_STEP, 3)
    head = torch.zeros(n, N_STEP)
    vel = torch.zeros(n, N_STEP, 2)
    for i, (p, h, v) in enumerate(tracks):
        pos[i, :, :2], head[i], vel[i] = p, h, v
    types = types or [0] * n
    shape = torch.tensor([[4.8, 2.0, 1.6] if t == 0 else [0.6, 0.6, 1.8] for t in types])
    role = torch.zeros(n, 3, dtype=torch.bool)
    for i, r in (roles or {}).items():
        role[i, r] = True
    data = {
        "agent": {
            "position": pos, "heading": head, "velocity": vel,
            "valid_mask": torch.ones(n, N_STEP, dtype=torch.bool),
            "shape": shape, "type": torch.tensor(types, dtype=torch.uint8), "role": role,
            "id": torch.arange(100, 100 + n),
        },
        "scenario_id": "test",
    }
    if map_save is not None:
        data.update(map_save)
    return data


def _lane_map(lines, pt_type=1, light=0):
    """Map of straight lanes, each split into 5 m segments of 3 points like
    the cached WOMD map; ``lines`` are (start, end) pairs."""
    segs, types, lights = [], [], []
    for (x0, y0), (x1, y1) in lines:
        p, q = torch.tensor([x0, y0], dtype=torch.float), torch.tensor([x1, y1], dtype=torch.float)
        n = max(int(torch.linalg.norm(q - p) / 5.0), 1)
        for k in range(n):
            a, b = p + (q - p) * k / n, p + (q - p) * (k + 1) / n
            segs.append(torch.stack([a, (a + b) / 2, b]))
            types.append(pt_type)
            lights.append(light)
    m = len(segs)
    return {
        "map_save": {"traj_pos": torch.stack(segs), "traj_theta": torch.zeros(m)},
        "pt_token": {"type": torch.tensor(types, dtype=torch.uint8),
                     "pl_type": torch.zeros(m, dtype=torch.uint8),
                     "light_type": torch.tensor(lights, dtype=torch.uint8), "num_nodes": m},
    }


STRAIGHT = [((-60, 0), (60, 0)), ((60, 3.5), (-60, 3.5))]
CROSS = STRAIGHT + [((0, -60), (0, 60)), ((3.5, 60), (3.5, -60))]


def test_following_in_one_lane():
    lead = _follow_path([[15, 0], [200, 0]], 10.0)
    follow = _follow_path([[0, 0], [200, 0]], 10.0)
    data = _scene([follow, lead], map_save=_lane_map(STRAIGHT))
    tag = tag_pair(data, 0, 1)
    assert tag.interaction_type == "following"
    assert tag.scene_type == "urban_road"
    assert tag.pet == pytest.approx(1.5, abs=0.15)  # the time headway
    # a 1.5 s headway holds over the whole clip
    assert tag.start_step == 0 and tag.end_step == N_STEP - 1


def test_queue_behind_a_stopped_car_is_following():
    stopped = _follow_path([[0, 0], [0.01, 0]], 0.0)
    approach = _follow_path([[-40, 0], [-7, 0]], 8.0)  # stops 7 m behind, a 2.2 m gap
    data = _scene([approach, stopped], map_save=_lane_map(STRAIGHT))
    assert classify_interaction(data, 0, 1)["interaction_type"] == "following"


def test_perpendicular_crossing_with_window_around_the_conflict():
    a = _follow_path([[-40, 0], [60, 0]], 10.0)  # at the origin at t = 4 s
    b = _follow_path([[0, -50], [0, 60]], 10.0)  # at the origin at t = 5 s
    data = _scene([a, b], map_save=_lane_map(CROSS, light=3))
    tag = tag_pair(data, 0, 1)
    assert tag.interaction_type == "crossing"
    assert tag.scene_type == "signalized_intersection"
    assert tag.pet == pytest.approx(1.0, abs=0.15)
    assert tag.first_through == "a"
    # active from approach_time before the later arrival, until both have passed
    assert 1.5 <= tag.start_s <= 2.5
    assert 4.0 <= tag.peak_s <= 5.0
    assert tag.end_s >= 5.0
    assert tag.end_s < 9.0  # and it does end


def test_unprotected_left_turn_against_oncoming_traffic():
    straight = _follow_path([[-50, 0], [60, 0]], 10.0)
    # oncoming in the opposite lane, turns left (towards -y) across the other's lane
    turn = [[40, 3.5], [8, 3.5]] + _arc((8, -4.5), 8.0, math.pi / 2, math.pi)[1:] + [[0, -60]]
    left = _follow_path(turn, 8.0)
    data = _scene([straight, left], map_save=_lane_map(CROSS))
    inter = classify_interaction(data, 0, 1)
    assert inter["maneuver_b"] == "left"
    assert inter["interaction_type"] == "left_turn_across_path"
    assert classify_scene(data, torch.tensor([0.0, 0.0]))["scene_type"] == "uncontrolled_intersection"


def test_lane_change_into_the_others_lane():
    keep = _follow_path([[0, 0], [200, 0]], 10.0)
    change = _follow_path([[8, 3.5], [20, 3.5], [45, 0], [250, 0]], 11.0)
    data = _scene([keep, change], map_save=_lane_map(STRAIGHT))
    inter = classify_interaction(data, 0, 1)
    assert inter["maneuver_b"] == "lane_change_right"
    assert inter["interaction_type"] == "lane_change"


def test_merge_when_one_agent_turns_into_the_lane():
    main = _follow_path([[-60, 0], [100, 0]], 10.0)
    # comes up from -y and turns right onto the road, ahead of the other car
    ramp = [[10, -40], [10, -8]] + _arc((18, -8), 8.0, math.pi, math.pi / 2)[1:] + [[150, 0]]
    turner = _follow_path(ramp, 8.0)
    data = _scene([main, turner], map_save=_lane_map(STRAIGHT))
    inter = classify_interaction(data, 0, 1)
    assert inter["maneuver_b"] == "right"
    assert inter["interaction_type"] == "merge"


def test_parallel_lanes_far_apart_do_not_interact():
    a = _follow_path([[0, 0], [200, 0]], 10.0)
    b = _follow_path([[0, 30], [200, 30]], 10.0)
    data = _scene([a, b])
    tag = tag_pair(data, 0, 1)
    assert tag.interaction_type == "none"
    assert tag.start_step == -1 and tag.duration_s == 0.0
    assert tag.scene_type == "no_lanes"


def test_side_by_side_is_adjacent_lane():
    a = _follow_path([[0, 0], [200, 0]], 10.0)
    b = _follow_path([[2, 3.5], [200, 3.5]], 10.0)
    assert classify_interaction(_scene([a, b]), 0, 1)["interaction_type"] == "adjacent_lane"


def test_pedestrian_crossing_in_front_of_a_vehicle():
    car = _follow_path([[-30, 0], [60, 0]], 8.0)  # at x=6 at 4.5 s
    ped = _follow_path([[6, -6], [6, 10]], 1.5)  # at y=0 at 4 s
    data = _scene([car, ped], types=[0, 1])
    inter = classify_interaction(data, 0, 1)
    assert inter["category"] == "vehicle_pedestrian"
    assert inter["interaction_type"] == "pedestrian_crossing"


def test_yielder_slows_down_for_the_one_passing_first():
    a = _follow_path([[-40, 0], [60, 0]], 10.0)  # at the origin at 4 s
    # b approaches at 8 m/s, waits 2 s just short of the crossing, then goes
    p1, h1, v1 = _follow_path([[0, -40], [0, -6]], 8.0)
    p2, h2, v2 = _follow_path([[0, -6], [0, 60]], 8.0, start_time=6.0)
    stop = int(round(34 / 8.0 / DT)) + 1
    b = (torch.cat([p1[:stop], p2[stop:]]), torch.cat([h1[:stop], h2[stop:]]), torch.cat([v1[:stop], v2[stop:]]))
    inter = classify_interaction(_scene([a, b]), 0, 1)
    assert inter["interaction_type"] == "crossing"
    assert inter["first_through"] == "a"
    assert inter["yielder"] == "b"


def test_highway_scene():
    lanes = _lane_map([((-60, 0), (60, 0)), ((-60, 3.7), (60, 3.7))], pt_type=0)
    data = _scene([_follow_path([[0, 0], [1, 0]], 0.0)], map_save=lanes)
    assert classify_scene(data, torch.tensor([0.0, 0.0]))["scene_type"] == "highway"


def test_stop_controlled_intersection():
    data = _scene([_follow_path([[0, 0], [1, 0]], 0.0)], map_save=_lane_map(CROSS, pt_type=2))
    scene = classify_scene(data, torch.tensor([0.0, 0.0]))
    assert scene["stop_sign"] and scene["scene_type"] == "stop_intersection"


def test_window_outside_the_radius_ignores_far_intersections():
    data = _scene([_follow_path([[0, 0], [1, 0]], 0.0)], map_save=_lane_map(CROSS))
    assert classify_scene(data, torch.tensor([0.0, 200.0]))["scene_type"] == "no_lanes"


def test_maneuvers():
    cfg = TagConfig()
    still = _scene([_follow_path([[0, 0], [0.5, 0]], 0.1)])
    assert agent_maneuver(still, 0, cfg)["maneuver"] == "stationary"
    uturn = _scene([_follow_path([[0, 0], [20, 0]] + _arc((20, 5), 5.0, -math.pi / 2, math.pi / 2)[1:] + [[0, 10]], 8.0)])
    assert agent_maneuver(uturn, 0, cfg)["maneuver"] == "u_turn"


def test_pair_selection_modes():
    a = _follow_path([[-40, 0], [60, 0]], 10.0)
    b = _follow_path([[0, -50], [0, 60]], 10.0)
    far = _follow_path([[0, 300], [100, 300]], 10.0)
    near = _follow_path([[-55, 0], [200, 0]], 10.0)  # 1.5 s behind a; 0.5 s after b at the origin
    interest = _scene([a, b, far, near], roles={0: [0], 1: [1], 2: [1], 3: [1]})
    # three objects of interest is not the interactive pair: auto falls back to the SDC
    assert interacting_pairs(interest, PAIRS_INTEREST) == []
    assert interacting_pairs(interest, PAIRS_AUTO)[0][2] == PAIRS_SDC

    pair = _scene([a, b, far, near], roles={0: [0], 1: [1], 3: [1]})
    assert interacting_pairs(pair, PAIRS_AUTO) == [(1, 3, PAIRS_INTEREST)]

    sdc = interacting_pairs(pair, PAIRS_SDC)
    assert sdc == [(0, 1, PAIRS_SDC)]  # PET 1.0 s beats the 1.5 s headway

    every = interacting_pairs(pair, PAIRS_ALL)
    assert {(x, y) for x, y, _ in every} == {(0, 1), (0, 3), (1, 3)}
    assert all(2 not in (x, y) for x, y, _ in every)


def test_window_ignores_unlogged_steps():
    a = _follow_path([[-40, 0], [60, 0]], 10.0)
    b = _follow_path([[0, -50], [0, 60]], 10.0)
    data = _scene([a, b])
    data["agent"]["valid_mask"][1, 50:] = False  # b's log ends at 5 s
    w = interaction_window(data, 0, 1)
    assert w["end_step"] <= 49

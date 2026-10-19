import math

import pytest
import torch

from tests.responsibility.conftest import FakeSMART

from src.responsibility.interaction import (
    InteractionConfig,
    geometric_criteria,
    geometric_interaction,
    influence_matrix,
    most_interactive_neighbour,
    select_interacting_neighbours,
    select_query_agents,
)
from src.responsibility.motion_set import MotionSetSynthesizer

NUM_HIST = 11
N_STEP = 41
NOW = NUM_HIST - 1
HORIZON = 30


class _Store(dict):
    pass


class _Data(dict):
    """Agents moving at constant velocity from given positions at the
    current step, laid out explicitly per test."""

    def __init__(self, starts, velocities, roles=None, types=None):
        super().__init__()
        n = len(starts)
        t = (torch.arange(N_STEP).float() - NOW) * 0.1  # seconds relative to now
        start = torch.tensor(starts, dtype=torch.float)
        vel = torch.tensor(velocities, dtype=torch.float)
        pos = start[:, None, :] + vel[:, None, :] * t[None, :, None]
        heading = torch.atan2(vel[:, 1], vel[:, 0])[:, None].expand(n, N_STEP).clone()
        self["agent"] = _Store(
            position=pos,
            heading=heading,
            velocity=vel[:, None, :].expand(n, N_STEP, 2).clone(),
            valid_mask=torch.ones(n, N_STEP, dtype=torch.bool),
            shape=torch.tensor([[4.8, 2.0, 1.5]]).expand(n, -1).clone(),
            type=torch.tensor(types if types is not None else [0] * n),
            role=torch.tensor(roles if roles is not None else [[False, False, False]] * n),
            id=torch.arange(100, 100 + n),
        )


def _geo(data, query, cands, **kw):
    return geometric_interaction(data, query, cands, NUM_HIST, HORIZON, InteractionConfig(**kw))


def test_same_lane_follower_far_behind_is_not_interacting():
    # both drive +x at 10 m/s, 30 m apart: they never close in
    data = _Data([[0, 0], [30, 0]], [[10, 0], [10, 0]])
    s = _geo(data, 0, [1])
    assert s["min_gap"][0] > 20
    assert math.isinf(s["ttc"][0])
    assert not geometric_criteria(s, InteractionConfig())[0]


def test_closing_in_gives_finite_ttc_and_small_gap():
    # agent 1 stopped 20 m ahead, agent 0 approaching at 10 m/s
    data = _Data([[0, 0], [20, 0]], [[10, 0], [0, 0]])
    s = _geo(data, 0, [1])
    assert 1.0 < s["ttc"][0] < 2.0  # bounding circles touch ~5 m before centres meet
    assert s["min_gap"][0] == 0.0  # footprints overlap within 3 s
    assert geometric_criteria(s, InteractionConfig())[0]


def test_crossing_paths_are_caught_by_pet_even_when_never_close_in_time():
    # agent 0 passes the origin at t=1 s going +x; agent 1 passes it at
    # t=2.5 s going +y. They are never within 10 m at the same instant.
    data = _Data([[-10, 0], [0, -25]], [[10, 0], [0, 10]])
    s = _geo(data, 0, [1], gap_threshold=0.0, ttc_threshold=0.0)
    assert s["min_gap"][0] > 5.0
    assert s["pet"][0] == pytest.approx(1.5, abs=0.25)
    assert geometric_criteria(s, InteractionConfig(gap_threshold=0.0, ttc_threshold=0.0))[0]


def test_parallel_adjacent_lanes_do_not_interact():
    data = _Data([[0, 0], [0, 3.7]], [[10, 0], [10, 0]])
    s = _geo(data, 0, [1], gap_threshold=0.5)
    assert s["min_gap"][0] > 0.5
    assert math.isinf(s["pet"][0]) or s["pet"][0] > 3.0


def test_radius_mode_is_the_plain_neighbourhood():
    data = _Data([[0, 0], [20, 0], [80, 0]], [[0, 0]] * 3)
    sel = select_interacting_neighbours(None, data, [0], NUM_HIST, HORIZON,
                                        InteractionConfig(mode="radius"), radius=30.0)
    assert sel.neighbours[0] == [1]


def test_geometric_mode_drops_non_interacting_and_ranks_by_gap():
    data = _Data(
        [[0, 0], [30, 0], [15, 0], [0, -15]],
        [[10, 0], [10, 0], [0, 0], [0, 10]],
    )  # 1: same speed 3 s ahead, never closes; 2: stopped ahead; 3: crosses 1.5 s later
    sel = select_interacting_neighbours(None, data, [0], NUM_HIST, HORIZON, InteractionConfig())
    assert 1 not in sel.neighbours[0]
    assert sel.neighbours[0][0] == 2  # overlaps -> gap 0 -> first
    assert 3 in sel.neighbours[0]
    assert sel.diagnostics[0][1]["selected"] == 0.0
    assert sel.diagnostics[0][2]["selected"] == 1.0


def test_max_neighbors_keeps_the_strongest():
    data = _Data([[0, 0], [8, 0], [14, 0], [20, 0]], [[10, 0], [0, 0], [0, 0], [0, 0]])
    sel = select_interacting_neighbours(None, data, [0], NUM_HIST, HORIZON, InteractionConfig(max_neighbors=1))
    assert sel.neighbours[0] == [1]


def test_prefilter_radius_bounds_the_candidates():
    data = _Data([[0, 0], [60, 0]], [[20, 0], [-20, 0]])  # head-on, but 60 m away now
    near = select_interacting_neighbours(None, data, [0], NUM_HIST, HORIZON, InteractionConfig(prefilter_radius=50))
    far = select_interacting_neighbours(None, data, [0], NUM_HIST, HORIZON, InteractionConfig(prefilter_radius=70))
    assert near.neighbours[0] == [] and far.neighbours[0] == [1]


@pytest.mark.parametrize("mode", ["radius", "geometric"])
def test_ooi_only_keeps_objects_of_interest(mode):
    # 1 (stopped ahead) and 3 (crossing) interact, only 3 is an object of interest
    roles = [[True, True, False], [False, False, False], [False, False, False], [False, True, False]]
    data = _Data(
        [[0, 0], [15, 0], [80, 0], [0, -15]],
        [[10, 0], [0, 0], [0, 0], [0, 10]],
        roles=roles,
    )
    every = select_interacting_neighbours(None, data, [0], NUM_HIST, HORIZON,
                                          InteractionConfig(mode=mode), radius=30.0)
    ooi = select_interacting_neighbours(None, data, [0], NUM_HIST, HORIZON,
                                        InteractionConfig(mode=mode, ooi_only=True), radius=30.0)
    assert set(every.neighbours[0]) == {1, 3}
    assert ooi.neighbours[0] == [3]


def test_query_modes():
    roles = [[True, False, False], [False, True, False], [False, False, True], [False, False, False]]
    data = _Data([[0, 0]] * 4, [[0, 0]] * 4, roles=roles, types=[0, 0, 1, 0])
    assert select_query_agents(data, NOW, "sdc") == [0]
    assert select_query_agents(data, NOW, "interest") == [1, 2]
    assert select_query_agents(data, NOW, "vehicles", agent_type=0) == [0, 1, 3]
    assert select_query_agents(data, NOW, "all") == [0, 1, 2, 3]
    data["agent"]["valid_mask"][3, NOW] = False
    assert 3 not in select_query_agents(data, NOW, "all")
    with pytest.raises(ValueError):
        select_query_agents(data, NOW, "ego")


def test_influence_matrix_shape_and_diagonal(sampling_scheme):
    synth = MotionSetSynthesizer(FakeSMART(n_agent=3), sampling_scheme, rollout_batch_size=16)
    data = _Data([[0, 0], [5, 0], [10, 0]], [[0, 0]] * 3)
    tok_map, tok_agent = synth.tokenize(data)

    m = influence_matrix(synth, tok_map, tok_agent, queried=[0, 1], candidates=[0, 1, 2])

    assert m.shape == (2, 3)
    assert m[0, 0] == 0.0 and m[1, 1] == 0.0  # removing yourself is not influence
    assert (m[0, 1:] > 0).all() and m[1, 0] > 0 and m[1, 2] > 0  # the stub reacts to removal


def test_influence_mode_needs_the_model():
    data = _Data([[0, 0], [5, 0]], [[0, 0]] * 2)
    with pytest.raises(ValueError, match="influence"):
        select_interacting_neighbours(None, data, [0], NUM_HIST, HORIZON, InteractionConfig(mode="influence"))


def test_influence_mode_adds_influential_agents_geometry_misses(sampling_scheme):
    # agent 1 is 40 m behind, same speed: geometrically inert, but the stub
    # model reacts to any removal, so influence selects it
    data = _Data([[0, 0], [-40, 0]], [[10, 0], [10, 0]])
    synth = MotionSetSynthesizer(FakeSMART(n_agent=2), sampling_scheme, rollout_batch_size=16)
    tok_map, tok_agent = synth.tokenize(data)

    geo = select_interacting_neighbours(synth, data, [0], NUM_HIST, HORIZON, InteractionConfig())
    infl = select_interacting_neighbours(synth, data, [0], NUM_HIST, HORIZON, InteractionConfig(mode="influence"),
                                         tokenized_map=tok_map, tokenized_agent=tok_agent)
    assert geo.neighbours[0] == []
    assert infl.neighbours[0] == [1]
    assert infl.diagnostics[0][1]["influence"] > 0


def test_crossing_two_and_a_half_seconds_apart_is_not_a_conflict():
    """PET beyond the 2 s threshold: the second vehicle reaches the crossing
    point well after the first has cleared it."""
    data = _Data([[0, 0], [0, -25]], [[10, 0], [0, 10]])
    s = _geo(data, 0, [1])
    assert s["pet"][0] > 2.0
    assert not geometric_criteria(s, InteractionConfig())[0]


@pytest.mark.parametrize(
    "direction, expected",
    [
        ("outgoing", {(100, "neighbours", 101)}),
        ("incoming", {(101, "toward_sdc", 100)}),
        ("both", {(100, "neighbours", 101), (101, "toward_sdc", 100)}),
    ],
)
def test_sdc_directions(sampling_scheme, direction, expected):
    from src.responsibility.dataset_builder import compute_scenario_responsibility
    from src.responsibility.metrics import ResponsibilityConfig

    # agent 0 is the SDC; agent 1 stopped just ahead interacts with it; agent 2 is far away
    roles = [[True, False, False], [False, False, False], [False, False, False]]
    data = _Data([[0, 0], [15, 0], [0, 45]], [[10, 0], [0, 0], [0, 0]], roles=roles)
    cfg = ResponsibilityConfig(num_historical_steps=NUM_HIST, n_safety_samples=2, n_courtesy_samples=2,
                               query_mode="sdc", query_direction=direction)
    synth = MotionSetSynthesizer(FakeSMART(n_agent=3), sampling_scheme, rollout_batch_size=32)

    obs = compute_scenario_responsibility(synth, data, cfg)

    got = {(o.agent_id, o.scope, n) for o in obs for n in o.per_neighbour}
    assert got == expected
    for o in obs:
        # evidence is kept for every candidate considered, selected or not,
        # so the thresholds can be calibrated after the fact
        selected = {n for n, d in o.interaction.items() if d["selected"] == 1.0}
        assert selected == set(o.per_neighbour)
        assert torch.isfinite(o.beta).all()


def test_side_by_side_in_adjacent_lanes_has_no_ttc():
    """Regression: bounding-circle TTC reported 0 for two cars driving side
    by side 3.5 m apart, whose footprints never touch."""
    data = _Data([[0, 0], [0, 3.5]], [[10, 0], [10, 0]])
    assert math.isinf(_geo(data, 0, [1])["ttc"][0])


def test_pet_counts_a_crossing_that_just_happened():
    # agent 1 crossed the origin 0.5 s ago going +y; agent 0 reaches it 0.5 s from now
    data = _Data([[-5, 0], [0, 5]], [[10, 0], [0, 10]])
    assert _geo(data, 0, [1])["pet"][0] == pytest.approx(1.0, abs=0.25)


def test_pet_ignores_a_conflict_entirely_in_the_past():
    # both passed the origin 0.5 s ago and are now driving apart
    data = _Data([[5, 0], [0, 5]], [[10, 0], [0, 10]])
    assert math.isinf(_geo(data, 0, [1])["pet"][0])


def test_most_interactive_neighbour_prefers_a_path_crossing_over_a_parallel_car():
    # 0 drives +x; 1 crosses 0's path at x=20 half a second after 0 passes;
    # 2 drives alongside in the next lane, close but never on 0's path
    data = _Data([[0, 0], [20, -25], [0, 4]], [[10, 0], [0, 10], [10, 0]])
    picked, scores = most_interactive_neighbour(data, 0, NOW, N_STEP - 1)
    assert picked == 1
    assert scores["pet"] < 1.0
    # without the crossing car, the closest one is the fallback
    data = _Data([[0, 0], [0, 4], [0, 12]], [[10, 0], [10, 0], [10, 0]])
    assert most_interactive_neighbour(data, 0, NOW, N_STEP - 1)[0] == 1


def test_most_interactive_neighbour_needs_someone_present():
    data = _Data([[0, 0], [5, 0]], [[10, 0], [10, 0]])
    data["agent"]["valid_mask"][1] = False
    assert most_interactive_neighbour(data, 0, NOW, N_STEP - 1) is None

import pickle
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch

from scripts.responsibility.report_figures import (
    Tracks,
    auc,
    label_windows,
    level_order,
    load_catk_windows,
    rule_reasons,
    speed_and_accel,
    time_to_collision,
    window_physics,
)

CFG = SimpleNamespace(prefilter=50.0, far=50.0, ttc_cap=10.0, lane_half_width=1.8, thw_min_speed=2.0, near=15.0,
                      rule_ttc=1.5, rule_thw=1.0, rule_accel=3.0)


def _following(gap=10.0, v_follow=10.0, v_lead=8.0, n_steps=91):
    """Two 4 m x 2 m cars on the x axis: the follower at 0, the leader
    ``gap`` m ahead bumper to bumper, both at constant speed."""
    t = np.arange(n_steps) * 0.1
    x = np.stack([v_follow * t, gap + 4.0 + v_lead * t])
    pos = np.stack([x, np.zeros_like(x)], axis=-1)
    vel = np.stack([np.full_like(x[0], v_follow), np.full_like(x[0], v_lead)])
    vel = np.stack([vel, np.zeros_like(vel)], axis=-1)
    return Tracks(["1", "2"], ["vehicle", "vehicle"], pos, vel, np.zeros((2, n_steps)), [4.0, 4.0], [2.0, 2.0],
                  np.ones((2, n_steps), dtype=bool))


def test_time_to_collision_head_on_and_separating():
    d = np.array([[10.0, 0.0], [10.0, 0.0]])
    v = np.array([[-5.0, 0.0], [5.0, 0.0]])
    t = time_to_collision(d, v, np.array([2.0, 2.0]), cap=10.0)
    assert np.isclose(t[0], 1.6)  # (10 - 2) / 5
    assert t[1] == 10.0
    assert time_to_collision(np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array(2.0), cap=10.0) == 0.0


def test_window_physics_closing_on_a_slower_car():
    tr = _following()
    speed, acc = speed_and_accel(tr)
    p = window_physics(tr, speed, acc, agent=0, step=10, horizon=20, cfg=CFG)
    # the gap closes at 2 m/s; the window's last step (t = 3 s) is the closest
    assert np.isclose(p["min_gap"], 10.0 - 2.0 * 3.0, atol=0.05)
    assert np.isclose(p["min_ttc"], 4.0 / 2.0, atol=0.05)
    assert np.isclose(p["min_thw"], 4.0 / 10.0, atol=0.01)
    assert p["own_brake"] == 0.0 and p["near_brake"] == 0.0
    assert rule_reasons(p, CFG) == ["headway"]


def test_level_order_ranks_calm_first_and_marks_elevated_levels():
    rows = ([{"safety": 0.0, "courtesy": 0.0, "level": 2}] * 20 + [{"safety": 1.0, "courtesy": 0.0, "level": 0}] * 5
            + [{"safety": 0.0, "courtesy": 1.0, "level": 1}] * 5)
    rank, aggressive = level_order(rows)
    assert rank[2] == 0
    assert sorted(aggressive) == [1, 2]


def test_threshold_verdicts_without_levels():
    rows = [{"safety": s, "courtesy": 0.0, "level": -1, "aggressive": None} for s in np.linspace(0.1, 1.0, 10)]
    assert label_windows(rows, rows, quantile=0.9) == "threshold"
    assert [r["aggressive"] for r in rows] == [0] * 9 + [1]


def test_auc():
    assert auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5


def test_catk_observations_load_without_the_model_code(tmp_path):
    """Observations pickled from catk's dataclass load through the stand-in."""
    module = types.ModuleType("src.responsibility.dataset_builder_stub")

    class ResponsibilityObservation:
        pass

    ResponsibilityObservation.__module__ = "src.responsibility.dataset_builder_stub"
    ResponsibilityObservation.__qualname__ = "ResponsibilityObservation"
    module.ResponsibilityObservation = ResponsibilityObservation
    sys.modules["src.responsibility.dataset_builder_stub"] = module
    try:
        obs = []
        for scope, k in (("neighbours", 10), ("toward_sdc", 10), ("neighbours", 15)):
            o = ResponsibilityObservation()
            o.__dict__.update(scenario_id="abc", agent_id=7, agent_idx=3, k=k, beta=torch.tensor([0.5, 0.2]),
                              level=1, scope=scope)
            obs.append(o)
        (tmp_path / "obs").mkdir()
        with open(tmp_path / "obs" / "abc.pkl", "wb") as f:
            pickle.dump(obs, f)
    finally:
        del sys.modules["src.responsibility.dataset_builder_stub"]
    rows = load_catk_windows(str(tmp_path))
    assert [r["step"] for r in rows] == [10, 15]
    assert rows[0]["agent"] == "7" and rows[0]["agent_index"] == 3 and rows[0]["level"] == 1
    assert np.isclose(rows[0]["safety"], 0.5) and np.isclose(rows[0]["courtesy"], 0.2)

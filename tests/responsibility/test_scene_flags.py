import numpy as np
import pytest

from src.responsibility.scene_flags import FlagConfig, ranking_metrics, scene_score, smooth, worst_window

KS = list(range(10, 75, 5))  # 13 windows, 0.5 s apart


def test_smooth_keeps_constants_and_the_mean_level_at_the_ends():
    x = np.full(9, 2.0)
    for smoother in ("moving_average", "gaussian", "ema"):
        assert np.allclose(smooth(x, 0.5, smoother, 1.0), 2.0)
    assert np.array_equal(smooth(np.arange(4.0), 0.5, "none", 1.0), np.arange(4.0))


def test_ema_is_causal():
    x = np.zeros(10)
    x[6] = 1.0
    assert np.all(smooth(x, 0.5, "ema", 1.0)[:6] == 0)
    assert np.all(smooth(x, 0.5, "gaussian", 0.5)[:6] > 0)


def test_worst_window_finds_the_sustained_stretch():
    x = np.array([0, 3, 0, 0, 2, 2, 2, 0], dtype=float)
    assert worst_window(x, 1) == (3.0, 1)
    assert worst_window(x, 3) == (2.0, 4)
    assert worst_window(x, 20)[0] == pytest.approx(x.mean())


def test_one_dangerous_stretch_flags_the_scene():
    beta = np.zeros((13, 2))
    beta[7:9, 0] = 1.5  # one cut-in, 1 s of high beta_s
    cfg = FlagConfig(smoother="none", window=1.0, safety_threshold=1.0)
    s = scene_score(KS, beta, cfg)
    assert s["flagged"] and s["safety"] == pytest.approx(1.5)
    assert s["safety_t"] == (4.5, 5.0)
    calm = scene_score(KS, np.zeros((13, 2)), cfg)
    assert not calm["flagged"]


def test_either_channel_uses_each_threshold():
    beta = np.zeros((13, 2))
    beta[3, 1] = 2.0
    cfg = FlagConfig(smoother="none", window=0.5, safety_threshold=0.5, courtesy_threshold=1.0)
    s = scene_score(KS, beta, cfg)
    assert s["either"] == pytest.approx(2.0) and s["flagged"]
    assert not scene_score(KS, beta, FlagConfig(smoother="none", window=0.5, channel="safety"))["flagged"]


def test_smoothing_suppresses_a_one_window_spike():
    beta = np.zeros((13, 2))
    beta[6, 0] = 1.0
    raw = scene_score(KS, beta, FlagConfig(smoother="none", window=0.5))["safety"]
    smoothed = scene_score(KS, beta, FlagConfig(smoother="gaussian", smooth_width=0.5, window=0.5))["safety"]
    assert raw == 1.0 and smoothed < 0.5


def test_ranking_metrics():
    m = ranking_metrics([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0], flagged=[True, False, False, False])
    assert m["auroc"] == 1.0 and m["precision@2"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 0.5

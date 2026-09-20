"""End-to-end smoke tests of the responsibility pipeline against the stub
decoder. ``metrics``, ``dataset_builder`` and ``selection`` import only
``motion_set`` from the SMART side, so with the stubbed sampler the whole
chain -- windowing, rollouts, both responsibility metrics, level
assignment, reward features and online selection -- runs without a
checkpoint, torch_geometric or a GPU. Values are meaningless (the stub is
not a traffic model); structure and bookkeeping are what is asserted.
"""

import numpy as np
import torch

from tests.responsibility.conftest import FakeSMART

from src.responsibility.dataset_builder import (
    assign_levels,
    build_reward_training_features,
    compute_scenario_responsibility,
    group_sequences,
    sequences_as_arrays,
    window_steps,
)
from src.responsibility.hmm import GaussianHMM
from src.responsibility.metrics import ResponsibilityConfig
from src.responsibility.motion_set import MotionSetSynthesizer
from src.responsibility.reward_model import train_reward_network
from src.responsibility.selection import predict_responsibility_aware
from tests.responsibility.test_windows import NUM_HIST, _Data


def _cfg(**kw):
    defaults = dict(
        num_historical_steps=NUM_HIST,
        metric_horizon=20,
        window_stride=20,
        n_safety_samples=4,
        n_courtesy_samples=2,
        neighbor_radius=30.0,
    )
    defaults.update(kw)
    return ResponsibilityConfig(**defaults)


def _synth(sampling_scheme, n_agent=3):
    return MotionSetSynthesizer(FakeSMART(n_agent=n_agent), sampling_scheme, rollout_batch_size=4)


def test_single_step_responsibility_runs_end_to_end(sampling_scheme):
    data, cfg = _Data(n_agent=3), _cfg()
    obs = compute_scenario_responsibility(_synth(sampling_scheme), data, cfg)

    assert len(obs) == 3  # one per vehicle
    for o in obs:
        assert o.k == NUM_HIST - 1
        assert o.beta.shape == (2,)
        assert torch.isfinite(o.beta).all()
        assert o.beta[1] >= 0.0, "courtesy is a KL and cannot be negative"


def test_windowed_responsibility_yields_a_sequence_per_agent(sampling_scheme):
    data, cfg = _Data(n_agent=3), _cfg()
    synth = _synth(sampling_scheme)
    steps = window_steps(data, cfg, synth, stride=cfg.window_stride)
    assert len(steps) >= 2

    obs = compute_scenario_responsibility(synth, data, cfg, steps=steps)

    assert len(obs) == len(steps) * 3
    grouped = group_sequences(obs)
    assert len(grouped) == 3  # one sequence per agent
    for sequence in grouped.values():
        assert [o.k for o in sequence] == steps

    arrays = sequences_as_arrays(grouped)
    assert all(a.shape == (len(steps), 2) for a in arrays)


def test_windowing_leaves_the_scenario_untouched(sampling_scheme):
    data, cfg = _Data(n_agent=3), _cfg()
    synth = _synth(sampling_scheme)
    before = {k: v.clone() for k, v in data["agent"].items()}

    compute_scenario_responsibility(
        synth, data, cfg, steps=window_steps(data, cfg, synth, stride=cfg.window_stride)
    )

    for key, value in before.items():
        assert torch.equal(data["agent"][key], value), f"{key} was mutated"


def test_reward_features_are_bucketed_by_level(sampling_scheme):
    data, cfg = _Data(n_agent=3), _cfg()
    synth = _synth(sampling_scheme)
    obs = compute_scenario_responsibility(synth, data, cfg)
    for i, o in enumerate(obs):
        o.scenario_id = "scene"
        o.level = i % 2

    n_aux = 3
    features = build_reward_training_features(
        synth, {"scene": data}, obs, cfg, n_aux=n_aux
    )

    assert set(features) <= {0, 1}
    for level, fs in features.items():
        n_real = fs.real.shape[0]
        assert fs.real.shape == (n_real, 4)
        assert fs.fake.shape == (n_real * n_aux, 4)
        assert torch.isfinite(fs.real).all() and torch.isfinite(fs.fake).all()


def test_online_selection_picks_a_candidate(sampling_scheme):
    data, cfg = _Data(n_agent=3), _cfg()
    synth = _synth(sampling_scheme)

    hmm = GaussianHMM(n_states=2, n_features=2, seed=0)
    rng = np.random.default_rng(0)
    hmm.fit([rng.normal(size=(6, 2)) for _ in range(5)], n_iter=5)

    real = torch.randn(16, 4) * 0.1
    fake = torch.randn(16, 4) * 0.1 + 3.0
    bundle, _ = train_reward_network(real, fake, hidden_dim=8, n_epochs=20)

    beta_history = torch.tensor([[0.2, 0.0], [0.4, 0.1]])
    out = predict_responsibility_aware(
        synth, data, 0, beta_history, hmm, {0: bundle, 1: bundle}, cfg, n_candidates=4
    )

    assert out["level"] in (0, 1)
    assert 0 <= out["best_idx"] < 4
    assert out["candidates"].shape[0] == 4
    assert out["features"].shape == (4, 4)
    assert out["selected_traj"].shape == out["candidates"].shape[1:]
    assert torch.equal(out["selected_traj"], out["candidates"][out["best_idx"]])
    assert out["horizon"] == cfg.metric_horizon


def test_level_assignment_fills_every_observation(sampling_scheme):
    data, cfg = _Data(n_agent=3), _cfg()
    synth = _synth(sampling_scheme)
    steps = window_steps(data, cfg, synth, stride=cfg.window_stride)
    obs = compute_scenario_responsibility(synth, data, cfg, steps=steps)

    grouped = group_sequences(obs)
    hmm = GaussianHMM(n_states=2, n_features=2, seed=0)
    hmm.fit(sequences_as_arrays(grouped), n_iter=5)
    assign_levels(grouped, hmm)

    assert all(o.level is not None for o in obs)
    assert all(0 <= o.level < 2 for o in obs)

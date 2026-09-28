"""Window-level planning: which rollouts responsibility needs, shared and packed."""

import torch

from tests.responsibility.conftest import FakeSMART
from tests.responsibility.test_windows import NUM_HIST, _Data

from src.responsibility.metrics import (
    ResponsibilityConfig,
    compute_agent_responsibility,
    plan_responsibility_jobs,
    responsibility_for_agents,
)
from src.responsibility.motion_set import MotionSetSynthesizer


def _cfg(**kw):
    base = dict(num_historical_steps=NUM_HIST, n_safety_samples=6, n_courtesy_samples=4)
    base.update(kw)
    return ResponsibilityConfig(**base)


def test_full_scene_rollouts_are_shared_between_safety_and_courtesy():
    # agents drive in a line 10 m apart: every adjacent pair interacts
    data, cfg = _Data(n_agent=2), _cfg()
    plan = plan_responsibility_jobs(data, queried=[0, 1], neighbors_of={0: [1], 1: [0]}, cfg=cfg)

    # one full-scene job per agent (sized for its larger use) + one "b without a" per pair
    assert set(plan.full_job) == {0, 1}
    assert set(plan.without_job) == {(0, 1), (1, 0)}
    assert len(plan.jobs) == 4
    for agent, j in plan.full_job.items():
        assert plan.jobs[j].free == (agent,) and plan.jobs[j].removed == ()
        assert plan.jobs[j].n_samples == max(cfg.n_safety_samples, cfg.n_courtesy_samples)
    for (a, b), j in plan.without_job.items():
        assert plan.jobs[j].free == (b,) and plan.jobs[j].removed == (a,)

    # unshared: 2 safety jobs + 2 x (with, without) courtesy jobs
    unshared = 2 * cfg.n_safety_samples + 2 * 2 * cfg.n_courtesy_samples
    assert plan.n_rollouts < unshared


def test_far_pairs_get_no_courtesy_rollouts():
    data = _Data(n_agent=3)  # agents 0 and 2 are 20 m apart
    cfg = _cfg(courtesy_skip_distance=15.0)
    plan = plan_responsibility_jobs(data, queried=[0], neighbors_of={0: [1, 2]}, cfg=cfg)

    assert plan.courtesy_targets[0] == [1]
    assert (0, 2) not in plan.without_job
    assert 2 not in plan.full_job


def test_agents_without_neighbours_cost_nothing():
    data, cfg = _Data(n_agent=2), _cfg()
    plan = plan_responsibility_jobs(data, queried=[0], neighbors_of={0: []}, cfg=cfg)
    assert plan.jobs == [] and plan.n_rollouts == 0


def test_window_results_cover_every_queried_agent(sampling_scheme):
    data, cfg = _Data(n_agent=3), _cfg(courtesy_skip_distance=15.0)
    model = FakeSMART(n_agent=3)
    synth = MotionSetSynthesizer(model, sampling_scheme, rollout_batch_size=64)
    tok_map, tok_agent = synth.tokenize(data)

    results, plan = responsibility_for_agents(
        synth, data, tok_map, tok_agent,
        queried=[0, 1, 2], neighbors_of={0: [1, 2], 1: [0, 2], 2: [0, 1]},
        all_agent_indices=[0, 1, 2], cfg=cfg,
    )

    assert set(results) == {0, 1, 2}
    for a, r in results.items():
        assert r["beta"].shape == (2,)
        assert torch.isfinite(r["beta"]).all()
        assert r["beta"][1] >= 0
        assert set(r["courtesy"]["per_neighbor"]) == {b for b in (0, 1, 2) if b != a}
    # skipped pair (0, 2) reports exactly zero courtesy
    assert results[0]["courtesy"]["per_neighbor"][2].item() == 0.0
    # everything fits in one packed call at this batch size
    assert len(model.encoder.seen_n_agent) == 1
    assert model.encoder.seen_n_agent[0] == plan.n_rollouts * 3


def test_single_agent_wrapper_goes_through_the_planner(sampling_scheme):
    data, cfg = _Data(n_agent=2), _cfg()
    synth = MotionSetSynthesizer(FakeSMART(n_agent=2), sampling_scheme, rollout_batch_size=64)
    tok_map, tok_agent = synth.tokenize(data)

    r = compute_agent_responsibility(synth, data, tok_map, tok_agent, 0, [1], [0, 1], cfg)

    assert r["beta"].shape == (2,)
    assert set(r["safety"]["per_neighbor"]) == {1}

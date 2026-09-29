"""Exact courtesy (Eq. 4) via the chain rule over replayed samples."""

import pytest
import torch

from tests.responsibility.conftest import FakeSMART
from tests.responsibility.test_windows import NUM_HIST, _Data

from src.responsibility.metrics import (
    ResponsibilityConfig,
    courtesy_responsibility_pair,
    exact_courtesy_from_logits,
    plan_responsibility_jobs,
    responsibility_for_agents,
)
from src.responsibility.motion_set import MotionSetSynthesizer, RolloutJob


def _cfg(**kw):
    base = dict(num_historical_steps=NUM_HIST, n_safety_samples=4, n_courtesy_samples=3,
                courtesy_estimator="exact")
    base.update(kw)
    return ResponsibilityConfig(**base)


def test_kl_of_identical_distributions_is_zero():
    logits = torch.randn(4, 3, 7)
    valid = torch.ones(4, 3, dtype=torch.bool)
    assert exact_courtesy_from_logits(logits, logits.clone(), valid) == pytest.approx(0.0, abs=1e-6)


def test_matches_a_hand_computed_chain_rule_kl():
    torch.manual_seed(0)
    p, q = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
    valid = torch.tensor([[True, True, True], [True, False, False]])

    got = exact_courtesy_from_logits(p, q, valid)

    lp, lq = torch.log_softmax(p, -1), torch.log_softmax(q, -1)
    step_kl = (lp.exp() * (lp - lq)).sum(-1)
    expected = (step_kl[0].sum() + step_kl[1, 0]) / 2  # sum over valid steps, mean over samples
    assert got == pytest.approx(float(expected), rel=1e-5)


def test_planner_draws_no_without_samples_for_the_exact_estimator():
    data = _Data(n_agent=2)
    plan = plan_responsibility_jobs(data, [0, 1], {0: [1], 1: [0]}, _cfg())
    assert plan.without_job == {}
    assert all(job.removed == () for job in plan.jobs)


def test_exact_courtesy_sees_influence_the_boltzmann_proxy_misses(full_sampling_scheme):
    """The stub's logits react to how many agents are present, so removing a
    changes b's distribution -- while b's sampled motions never collide with
    a, which leaves the Boltzmann proxy at exactly zero."""
    data = _Data(n_agent=2)
    synth = MotionSetSynthesizer(FakeSMART(n_agent=2), full_sampling_scheme, rollout_batch_size=32)
    tok_map, tok_agent = synth.tokenize(data)

    exact = courtesy_responsibility_pair(synth, data, tok_map, tok_agent, 0, 1, [0, 1], _cfg())
    proxy = courtesy_responsibility_pair(synth, data, tok_map, tok_agent, 0, 1, [0, 1],
                                         _cfg(courtesy_estimator="boltzmann"))

    assert exact.item() > 0.0
    assert proxy.item() == 0.0


def test_window_pass_matches_the_per_pair_exact_estimate(full_sampling_scheme):
    data = _Data(n_agent=3)
    synth = MotionSetSynthesizer(FakeSMART(n_agent=3), full_sampling_scheme, rollout_batch_size=64)
    tok_map, tok_agent = synth.tokenize(data)
    cfg = _cfg(courtesy_skip_distance=None)

    results, _ = responsibility_for_agents(
        synth, data, tok_map, tok_agent, [0], {0: [1, 2]}, [0, 1, 2], cfg
    )
    for b in (1, 2):
        single = courtesy_responsibility_pair(synth, data, tok_map, tok_agent, 0, b, [0, 1, 2], cfg)
        # the stub is deterministic, so both routes see identical samples
        assert results[0]["courtesy"]["per_neighbor"][b].item() == pytest.approx(single.item(), rel=1e-5)
    assert results[0]["beta"][1] > 0


def test_replaying_in_the_same_context_gives_zero_kl(full_sampling_scheme):
    synth = MotionSetSynthesizer(FakeSMART(n_agent=3), full_sampling_scheme, rollout_batch_size=32)
    tok_map, tok_agent = synth.tokenize(_Data(n_agent=3))
    (free,) = synth.run_jobs(tok_map, tok_agent, [RolloutJob(free=(1,), n_samples=3)], keep_logits=True)
    tokens = free["next_token_idx"][:, 0]
    (replay,) = synth.run_jobs(
        tok_map, tok_agent, [RolloutJob(free=(), n_samples=3, outputs=(1,), replay={1: tokens})], keep_logits=True
    )
    assert torch.equal(replay["next_token_idx"][:, 0], tokens)
    kl = exact_courtesy_from_logits(free["next_token_logits"][:, 0], replay["next_token_logits"][:, 0],
                                    free["next_token_valid"][:, 0])
    assert kl.item() == pytest.approx(0.0, abs=1e-6)


def test_exact_estimator_refuses_simulated_neighbours(full_sampling_scheme):
    data = _Data(n_agent=2)
    synth = MotionSetSynthesizer(FakeSMART(n_agent=2), full_sampling_scheme)
    tok_map, tok_agent = synth.tokenize(data)
    with pytest.raises(ValueError, match="simulated"):
        responsibility_for_agents(synth, data, tok_map, tok_agent, [0], {0: [1]}, [0, 1],
                                  _cfg(neighbor_future="simulated"))


@pytest.mark.parametrize("num_k, temp", [(4, 1.0), (32, 0.7)])
def test_exact_estimator_refuses_truncated_or_tempered_sampling(num_k, temp):
    from omegaconf import OmegaConf

    scheme = OmegaConf.create({"criterium": "topk_prob", "num_k": num_k, "temp": temp})
    data = _Data(n_agent=2)
    synth = MotionSetSynthesizer(FakeSMART(n_agent=2), scheme)
    tok_map, tok_agent = synth.tokenize(data)
    with pytest.raises(ValueError, match="full token distribution"):
        responsibility_for_agents(synth, data, tok_map, tok_agent, [0], {0: [1]}, [0, 1],
                                  _cfg(courtesy_skip_distance=None))
    with pytest.raises(ValueError, match="full token distribution"):
        courtesy_responsibility_pair(synth, data, tok_map, tok_agent, 0, 1, [0, 1],
                                     _cfg(courtesy_skip_distance=None))
    # the Boltzmann proxy does not care how the samples were drawn
    responsibility_for_agents(synth, data, tok_map, tok_agent, [0], {0: [1]}, [0, 1],
                              _cfg(courtesy_estimator="boltzmann"))

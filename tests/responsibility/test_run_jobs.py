"""The packed rollout engine: many heterogeneous jobs per model call."""

import torch

from tests.responsibility.conftest import (
    FIRST_N_STEP,
    FREE_TOKEN,
    N_FUTURE_2HZ,
    N_TOKEN,
    FakeSMART,
    make_tokenized_agent,
    make_tokenized_map,
)

from src.responsibility.motion_set import MotionSetSynthesizer, RolloutJob, take_samples

N_AGENT = 4
GT = torch.arange(FIRST_N_STEP, FIRST_N_STEP + N_FUTURE_2HZ)


def _setup(sampling_scheme, batch):
    model = FakeSMART(n_agent=N_AGENT)
    synth = MotionSetSynthesizer(model, sampling_scheme, rollout_batch_size=batch)
    return model, synth, make_tokenized_map(), make_tokenized_agent(N_AGENT)


def test_jobs_are_packed_across_calls(sampling_scheme):
    model, synth, tok_map, tok_agent = _setup(sampling_scheme, batch=4)
    jobs = [RolloutJob(free=(1,), n_samples=3), RolloutJob(free=(2,), n_samples=2)]

    results = synth.run_jobs(tok_map, tok_agent, jobs)

    # 5 copies at 4 per call -> one call of 4 copies, one of 1
    assert model.encoder.seen_n_agent == [4 * N_AGENT, N_AGENT]
    assert results[0]["pred_traj_10hz"].shape[:2] == (3, 1)
    assert results[1]["pred_traj_10hz"].shape[:2] == (2, 1)


def test_each_copy_gets_its_own_free_and_removed_agents(sampling_scheme):
    model, synth, tok_map, tok_agent = _setup(sampling_scheme, batch=8)
    jobs = [
        RolloutJob(free=(1,), n_samples=2, outputs=(0, 1, 2)),
        RolloutJob(free=(2,), n_samples=2, removed=(0,), outputs=(1, 2)),
    ]

    a, b = synth.run_jobs(tok_map, tok_agent, jobs)

    # one call holds all four copies; only job b's copies drop agent 0
    (seen,) = model.encoder.seen_valid_masks
    blocks = seen.view(4, N_AGENT, -1)
    assert blocks[0, 0].all() and blocks[1, 0].all()
    assert not blocks[2, 0].any() and not blocks[3, 0].any()

    free_tok = torch.full((N_FUTURE_2HZ,), FREE_TOKEN)
    for s in range(2):
        # job a outputs agents (0, 1, 2) with agent 1 free
        assert torch.equal(a["next_token_idx"][s, 0], GT)
        assert torch.equal(a["next_token_idx"][s, 1], free_tok)
        assert torch.equal(a["next_token_idx"][s, 2], GT)
        # job b outputs agents (1, 2) with agent 2 free: agent 1 is forced there
        assert torch.equal(b["next_token_idx"][s, 0], GT)
        assert torch.equal(b["next_token_idx"][s, 1], free_tok)


def test_packing_matches_separate_calls(sampling_scheme):
    _, packed, tok_map, tok_agent = _setup(sampling_scheme, batch=16)
    _, single, _, _ = _setup(sampling_scheme, batch=1)
    jobs = [
        RolloutJob(free=(0,), n_samples=2),
        RolloutJob(free=(3,), n_samples=3, removed=(1,)),
    ]

    together = packed.run_jobs(tok_map, tok_agent, jobs)
    apart = [single.run_jobs(tok_map, tok_agent, [job])[0] for job in jobs]

    for x, y in zip(together, apart):
        for key in ("pred_traj_10hz", "next_token_idx", "next_token_valid", "token_log_prob"):
            assert torch.equal(x[key], y[key]), key


def test_replay_forces_the_given_token_sequences(sampling_scheme):
    _, synth, tok_map, tok_agent = _setup(sampling_scheme, batch=8)
    n = 3
    tokens = torch.stack([torch.full((N_FUTURE_2HZ,), 5 + s) for s in range(n)])  # distinct per sample

    (res,) = synth.run_jobs(
        tok_map, tok_agent, [RolloutJob(free=(), n_samples=n, outputs=(1,), replay={1: tokens})]
    )

    assert torch.equal(res["next_token_idx"][:, 0], tokens)


def test_replay_overrides_missing_ground_truth(sampling_scheme):
    """A replayed agent must be forced even where it has no logged token."""
    _, synth, tok_map, tok_agent = _setup(sampling_scheme, batch=8)
    tok_agent["valid_mask"][1, 6:] = False
    tokens = torch.full((1, N_FUTURE_2HZ), 7)

    (res,) = synth.run_jobs(
        tok_map, tok_agent, [RolloutJob(free=(), n_samples=1, outputs=(1,), replay={1: tokens})]
    )

    assert torch.equal(res["next_token_idx"][0, 0], tokens[0])
    assert not tok_agent["valid_mask"][1, 6:].any(), "caller's tokenized_agent was mutated"


def test_token_log_prob_matches_the_logits(sampling_scheme):
    _, synth, tok_map, tok_agent = _setup(sampling_scheme, batch=8)
    (res,) = synth.run_jobs(
        tok_map, tok_agent, [RolloutJob(free=(1,), n_samples=2, outputs=(0, 1))], keep_logits=True
    )

    expected = torch.log_softmax(res["next_token_logits"], dim=-1).gather(
        -1, res["next_token_idx"].unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(res["token_log_prob"], expected)
    assert res["token_log_prob"].shape == (2, 2, N_FUTURE_2HZ)


def test_logits_are_dropped_unless_requested(sampling_scheme):
    _, synth, tok_map, tok_agent = _setup(sampling_scheme, batch=8)
    (res,) = synth.run_jobs(tok_map, tok_agent, [RolloutJob(free=(1,), n_samples=1)])
    assert "next_token_logits" not in res
    assert "token_log_prob" in res


def test_take_samples_slices_the_sample_axis(sampling_scheme):
    _, synth, tok_map, tok_agent = _setup(sampling_scheme, batch=8)
    (res,) = synth.run_jobs(tok_map, tok_agent, [RolloutJob(free=(1,), n_samples=5)])
    head = take_samples(res, 2)
    assert head["pred_traj_10hz"].shape[0] == 2
    assert torch.equal(head["output_agent_indices"], res["output_agent_indices"])


def test_auto_batch_size_on_cpu(sampling_scheme):
    synth = MotionSetSynthesizer(FakeSMART(n_agent=2), sampling_scheme)
    assert synth.rollout_batch_size == MotionSetSynthesizer.DEFAULT_BATCH_CPU


def test_map_is_encoded_once_per_call_not_per_copy(sampling_scheme):
    model, synth, tok_map, tok_agent = _setup(sampling_scheme, batch=4)
    n_pl = tok_map["position"].shape[0]
    jobs = [RolloutJob(free=(1,), n_samples=6), RolloutJob(free=(2,), n_samples=3, removed=(0,))]

    synth.run_jobs(tok_map, tok_agent, jobs)

    # 9 copies over 3 calls, but the map is encoded once, at its original size
    assert len(model.encoder.seen_n_agent) == 3
    assert model.encoder.map_encoder.calls == 1
    assert model.encoder.map_encoder.seen_n_pl == [n_pl]

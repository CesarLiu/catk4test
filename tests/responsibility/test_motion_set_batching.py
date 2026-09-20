import contextlib

import pytest
import torch

from tests.responsibility.conftest import (
    AGENT_DECODER_STUB,
    FIRST_N_STEP,
    FREE_TOKEN,
    N_FUTURE_2HZ,
    N_TOKEN,
    SHIFT,
    FakeSMART,
    make_tokenized_agent,
    make_tokenized_map,
    row_indexed_sampler,
)

from src.responsibility.motion_set import (
    MotionSetSynthesizer,
    replicate_scene,
    temporary_rollout_horizon,
)


@contextlib.contextmanager
def _sampler(fn):
    original = AGENT_DECODER_STUB.sample_next_token_traj
    AGENT_DECODER_STUB.sample_next_token_traj = fn
    try:
        yield
    finally:
        AGENT_DECODER_STUB.sample_next_token_traj = original


def test_replicate_scene_offsets_batch_and_repeats_nodes():
    n_agent, n_pl, n_copies = 4, 6, 3
    tok_map, tok_agent = make_tokenized_map(n_pl), make_tokenized_agent(n_agent)

    rep_map, rep_agent = replicate_scene(tok_map, tok_agent, n_copies)

    assert rep_agent["num_graphs"] == n_copies
    assert rep_agent["valid_mask"].shape[0] == n_agent * n_copies
    assert rep_map["position"].shape[0] == n_pl * n_copies
    assert torch.equal(
        rep_agent["batch"], torch.arange(n_copies).repeat_interleave(n_agent)
    )
    assert torch.equal(rep_map["batch"], torch.arange(n_copies).repeat_interleave(n_pl))
    # each copy carries the original per-agent tensors
    for c in range(n_copies):
        block = rep_agent["gt_idx"][c * n_agent : (c + 1) * n_agent]
        assert torch.equal(block, tok_agent["gt_idx"])


def test_replicate_scene_leaves_shared_vocabularies_alone():
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(4)
    rep_map, rep_agent = replicate_scene(tok_map, tok_agent, 3)

    assert rep_agent["trajectory_token_veh"].shape[0] == N_TOKEN
    assert rep_map["token_traj_src"].shape[0] == N_TOKEN


def test_replicate_scene_is_a_noop_for_one_copy():
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(3)
    out_map, out_agent = replicate_scene(tok_map, tok_agent, 1)
    assert out_map is tok_map and out_agent is tok_agent


def test_batched_rollouts_match_requested_sample_count(sampling_scheme):
    model = FakeSMART(n_agent=4)
    synth = MotionSetSynthesizer(model, sampling_scheme, rollout_batch_size=2)
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(4)

    out = synth.sample_rollouts(tok_map, tok_agent, free_agent_indices=[1], n_samples=5)

    assert out["pred_traj_10hz"].shape[0] == 5
    # 5 samples at batch size 2 -> chunks of 2, 2, 1
    assert len(model.encoder.seen_n_agent) == 3
    assert model.encoder.seen_n_agent == [8, 8, 4]


def test_batched_rollouts_gather_the_right_rows(sampling_scheme):
    """Each replicated copy must be read back from its own block of rows."""
    n_agent, free_idx, n_samples = 4, 2, 3
    model = FakeSMART(n_agent=n_agent)
    synth = MotionSetSynthesizer(model, sampling_scheme, rollout_batch_size=n_samples)
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(n_agent)

    with _sampler(row_indexed_sampler):
        out = synth.sample_rollouts(
            tok_map, tok_agent, free_agent_indices=[free_idx], n_samples=n_samples
        )

    realized = out["next_token_idx"][:, 0, 0]  # first future step of each sample
    expected = torch.tensor([(c * n_agent + free_idx) % N_TOKEN for c in range(n_samples)])
    assert torch.equal(realized, expected)


def test_forcing_still_applies_in_every_replicated_copy(sampling_scheme):
    n_agent, n_samples = 3, 4
    model = FakeSMART(n_agent=n_agent)
    synth = MotionSetSynthesizer(model, sampling_scheme, rollout_batch_size=n_samples)
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(n_agent)

    out = synth.sample_rollouts(
        tok_map,
        tok_agent,
        free_agent_indices=[0],
        n_samples=n_samples,
        output_agent_indices=[0, 1, 2],
    )

    expected_gt = torch.arange(FIRST_N_STEP, FIRST_N_STEP + N_FUTURE_2HZ)
    for sample in range(n_samples):
        assert torch.equal(out["next_token_idx"][sample, 0], torch.full((N_FUTURE_2HZ,), FREE_TOKEN))
        for forced_slot in (1, 2):
            assert torch.equal(out["next_token_idx"][sample, forced_slot], expected_gt)


def test_batching_does_not_change_results(sampling_scheme):
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(4)

    unbatched = MotionSetSynthesizer(FakeSMART(n_agent=4), sampling_scheme, rollout_batch_size=1)
    batched = MotionSetSynthesizer(FakeSMART(n_agent=4), sampling_scheme, rollout_batch_size=4)

    a = unbatched.sample_rollouts(tok_map, tok_agent, free_agent_indices=[1], n_samples=4)
    b = batched.sample_rollouts(tok_map, tok_agent, free_agent_indices=[1], n_samples=4)

    assert torch.equal(a["next_token_idx"], b["next_token_idx"])
    assert torch.equal(a["pred_traj_10hz"], b["pred_traj_10hz"])


def test_temporary_rollout_horizon_shortens_and_restores(sampling_scheme):
    model = FakeSMART(n_agent=3)
    synth = MotionSetSynthesizer(model, sampling_scheme)
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(3)
    full = model.encoder.num_future_steps

    out = synth.sample_rollouts(
        tok_map, tok_agent, free_agent_indices=[0], n_samples=2, rollout_horizon=20
    )

    assert out["pred_traj_10hz"].shape[2] == 20
    assert out["next_token_idx"].shape[2] == 20 // SHIFT
    assert out["next_token_valid"].shape[2] == 20 // SHIFT
    assert model.encoder.num_future_steps == full


def test_rollout_horizon_must_be_a_multiple_of_the_token_shift(sampling_scheme):
    model = FakeSMART(n_agent=2)
    with pytest.raises(ValueError):
        with temporary_rollout_horizon(model.encoder, 7):
            pass
    assert model.encoder.num_future_steps == N_FUTURE_2HZ * SHIFT


def test_rollout_horizon_is_restored_after_an_exception():
    model = FakeSMART(n_agent=2)
    full = model.encoder.num_future_steps
    with pytest.raises(RuntimeError):
        with temporary_rollout_horizon(model.encoder, 10):
            raise RuntimeError("boom")
    assert model.encoder.num_future_steps == full


def test_rollout_horizon_longer_than_the_model_is_capped(sampling_scheme):
    model = FakeSMART(n_agent=2)
    full = model.encoder.num_future_steps
    with temporary_rollout_horizon(model.encoder, full * 2):
        assert model.encoder.num_future_steps == full

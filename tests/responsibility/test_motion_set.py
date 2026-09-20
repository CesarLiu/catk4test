import pytest
import torch

from tests.responsibility.conftest import (
    AGENT_DECODER_STUB,
    FIRST_N_STEP,
    FREE_TOKEN,
    N_FUTURE_2HZ,
    N_STEP_TOKEN,
    FakeSMART,
    make_tokenized_agent,
    make_tokenized_map,
)

from src.responsibility.motion_set import (
    MotionSetSynthesizer,
    find_neighbors,
    force_ground_truth_tokens,
    remove_agents,
)


def _free_mask(n_agent, free_idx):
    mask = torch.zeros(n_agent, dtype=torch.bool)
    mask[free_idx] = True
    return mask


def test_forcing_replays_ground_truth_tokens_for_non_free_agents(sampling_scheme):
    model = FakeSMART(n_agent=4)
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(4)
    free_mask = _free_mask(4, 1)

    with force_ground_truth_tokens(tok_agent, free_mask):
        out = model.encoder.inference(tok_map, tok_agent, sampling_scheme)

    future_idx = out["pred_idx"][:, FIRST_N_STEP : FIRST_N_STEP + N_FUTURE_2HZ]
    expected_gt = torch.arange(FIRST_N_STEP, FIRST_N_STEP + N_FUTURE_2HZ)

    for a in [0, 2, 3]:
        assert torch.equal(future_idx[a], expected_gt), f"agent {a} was not teacher-forced"
    assert torch.equal(
        future_idx[1], torch.full((N_FUTURE_2HZ,), FREE_TOKEN)
    ), "the free agent must keep the sampler's own tokens"


def test_forcing_step_counter_is_per_inference_call(sampling_scheme):
    """Regression: the step counter must restart at n_step=2 for every
    ``.inference()`` call. Reusing one sampler across calls walked it past
    the 18-step gt_idx array and raised IndexError on the second call.
    ``rollout_batch_size=1`` forces one call per sample, which is exactly
    the configuration that used to break."""
    model = FakeSMART(n_agent=3)
    synth = MotionSetSynthesizer(model, sampling_scheme, rollout_batch_size=1)
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(3)

    rollouts = synth.sample_rollouts(
        tok_map, tok_agent, free_agent_indices=[0], n_samples=3, output_agent_indices=[0, 1]
    )

    assert len(model.encoder.seen_valid_masks) == 3, "expected one inference call per sample"
    expected_gt = torch.arange(FIRST_N_STEP, FIRST_N_STEP + N_FUTURE_2HZ)
    for sample in range(3):
        assert torch.equal(rollouts["next_token_idx"][sample, 1], expected_gt)


def test_forcing_skips_steps_without_valid_ground_truth(sampling_scheme):
    model = FakeSMART(n_agent=2)
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(2)
    # agent 0 has no valid GT token at n_step=5, so it cannot be forced there
    tok_agent["valid_mask"][0, 5] = False
    free_mask = _free_mask(2, 1)

    with force_ground_truth_tokens(tok_agent, free_mask):
        out = model.encoder.inference(tok_map, tok_agent, sampling_scheme)

    assert out["pred_idx"][0, 5].item() == FREE_TOKEN
    assert out["pred_idx"][0, 4].item() == 4  # neighbouring steps still forced
    assert out["pred_idx"][0, 6].item() == 6


def test_sampler_is_restored_after_context(sampling_scheme):
    original = AGENT_DECODER_STUB.sample_next_token_traj
    tok_agent = make_tokenized_agent(2)

    with force_ground_truth_tokens(tok_agent, _free_mask(2, 0)):
        assert AGENT_DECODER_STUB.sample_next_token_traj is not original
    assert AGENT_DECODER_STUB.sample_next_token_traj is original


def test_sampler_is_restored_after_exception(sampling_scheme):
    original = AGENT_DECODER_STUB.sample_next_token_traj
    tok_agent = make_tokenized_agent(2)

    with pytest.raises(RuntimeError):
        with force_ground_truth_tokens(tok_agent, _free_mask(2, 0)):
            raise RuntimeError("boom")
    assert AGENT_DECODER_STUB.sample_next_token_traj is original


def test_remove_agents_zeroes_validity_without_mutating_input():
    tok_agent = make_tokenized_agent(3)
    before = tok_agent["valid_mask"].clone()

    stripped = remove_agents(tok_agent, [1])

    assert not stripped["valid_mask"][1].any()
    assert stripped["valid_mask"][0].all() and stripped["valid_mask"][2].all()
    assert torch.equal(tok_agent["valid_mask"], before), "input dict was mutated"
    assert stripped["gt_idx"] is tok_agent["gt_idx"], "untouched fields should be shared"


def test_sample_rollouts_removes_agent_from_the_scene(sampling_scheme):
    model = FakeSMART(n_agent=4)
    synth = MotionSetSynthesizer(model, sampling_scheme)
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(4)

    synth.sample_rollouts(
        tok_map,
        tok_agent,
        free_agent_indices=[2],
        n_samples=2,
        remove_agent_indices=[0],
    )

    for seen in model.encoder.seen_valid_masks:
        assert not seen[0].any(), "removed agent must be invalid in every rollout"
        assert seen[2].all(), "queried agent must stay valid"
    assert tok_agent["valid_mask"].all(), "caller's tokenized_agent was mutated"


def test_sample_rollouts_output_shapes(sampling_scheme):
    model = FakeSMART(n_agent=5)
    synth = MotionSetSynthesizer(model, sampling_scheme)
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(5)
    n_samples = 4

    out = synth.sample_rollouts(
        tok_map, tok_agent, free_agent_indices=[1, 3], n_samples=n_samples
    )

    horizon = N_FUTURE_2HZ * 5
    assert out["pred_traj_10hz"].shape == (n_samples, 2, horizon, 2)
    assert out["pred_head_10hz"].shape == (n_samples, 2, horizon)
    assert out["next_token_logits"].shape[:2] == (n_samples, 2)
    assert out["next_token_idx"].shape == (n_samples, 2, N_FUTURE_2HZ)
    assert out["next_token_valid"].shape == (n_samples, 2, N_FUTURE_2HZ)
    assert out["next_token_logits"].shape[2] == N_FUTURE_2HZ


def test_sample_rollouts_free_agents_are_not_forced(sampling_scheme):
    model = FakeSMART(n_agent=3)
    synth = MotionSetSynthesizer(model, sampling_scheme)
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(3)

    out = synth.sample_rollouts(tok_map, tok_agent, free_agent_indices=[2], n_samples=2)

    assert torch.equal(
        out["next_token_idx"], torch.full((2, 1, N_FUTURE_2HZ), FREE_TOKEN)
    )


def test_eval_mode_is_required(sampling_scheme):
    model = FakeSMART(n_agent=2)
    model.training = True
    with pytest.raises(ValueError):
        MotionSetSynthesizer(model, sampling_scheme)


def test_find_neighbors_by_radius():
    n_agent, n_step = 4, N_STEP_TOKEN
    position = torch.zeros(n_agent, n_step, 2)
    position[1, :, 0] = 5.0
    position[2, :, 0] = 50.0
    position[3, :, 0] = 5.0
    valid = torch.ones(n_agent, n_step, dtype=torch.bool)
    valid[3] = False

    mask = find_neighbors(position, valid, query_idx=0, at_step=10, radius=30.0)

    assert mask[1] and not mask[0] and not mask[2] and not mask[3]

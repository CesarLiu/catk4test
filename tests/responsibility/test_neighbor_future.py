"""The neighbor_future switch: how much of the other agents' future the
queried agent is allowed to condition on while its alternatives are drawn.
"""

import pytest
import torch

from tests.responsibility.conftest import (
    FIRST_N_STEP,
    FREE_TOKEN,
    N_FUTURE_2HZ,
    FakeSMART,
    make_tokenized_agent,
    make_tokenized_map,
)

from src.responsibility.motion_set import (
    NEIGHBOR_FUTURE_HIDDEN,
    NEIGHBOR_FUTURE_LOGGED,
    NEIGHBOR_FUTURE_MODES,
    NEIGHBOR_FUTURE_SIMULATED,
    MotionSetSynthesizer,
    hide_agents_after_current_step,
)

FREE_IDX = 1
N_AGENT = 3


def _setup(sampling_scheme, n_agent=N_AGENT):
    model = FakeSMART(n_agent=n_agent)
    synth = MotionSetSynthesizer(model, sampling_scheme, rollout_batch_size=1)
    return model, synth, make_tokenized_map(), make_tokenized_agent(n_agent)


def _run(sampling_scheme, mode, n_samples=1):
    model, synth, tok_map, tok_agent = _setup(sampling_scheme)
    out = synth.sample_rollouts(
        tok_map,
        tok_agent,
        free_agent_indices=[FREE_IDX],
        n_samples=n_samples,
        output_agent_indices=list(range(N_AGENT)),
        neighbor_future=mode,
    )
    return model, out


def test_logged_keeps_neighbours_visible_all_the_way(sampling_scheme):
    model, _ = _run(sampling_scheme, NEIGHBOR_FUTURE_LOGGED)

    assert len(model.encoder.seen_a2a_masks) == N_FUTURE_2HZ
    for mask in model.encoder.seen_a2a_masks:
        assert mask.all(), "no agent should be hidden in logged mode"


def test_hidden_cuts_neighbours_after_the_current_step(sampling_scheme):
    model, _ = _run(sampling_scheme, NEIGHBOR_FUTURE_HIDDEN)

    masks = model.encoder.seen_a2a_masks
    assert masks[0].all(), "history steps must stay visible"
    for mask in masks[1:]:
        assert mask[FREE_IDX].all(), "the queried agent stays in the graph"
        for other in (0, 2):
            assert not mask[other].any(), "neighbours must drop out after the current step"


def test_hidden_still_replays_the_logged_neighbour_trajectories(sampling_scheme):
    """Hiding changes what the queried agent *sees*, not what the
    neighbours *do* -- they still follow their logs."""
    _, out = _run(sampling_scheme, NEIGHBOR_FUTURE_HIDDEN)

    expected_gt = torch.arange(FIRST_N_STEP, FIRST_N_STEP + N_FUTURE_2HZ)
    for other in (0, 2):
        assert torch.equal(out["next_token_idx"][0, other], expected_gt)
    assert torch.equal(
        out["next_token_idx"][0, FREE_IDX], torch.full((N_FUTURE_2HZ,), FREE_TOKEN)
    )


def test_simulated_forces_nobody(sampling_scheme):
    _, out = _run(sampling_scheme, NEIGHBOR_FUTURE_SIMULATED)

    # with no teacher forcing every agent takes the sampler's own token
    assert torch.equal(
        out["next_token_idx"], torch.full((1, N_AGENT, N_FUTURE_2HZ), FREE_TOKEN)
    )


def test_simulated_leaves_the_graph_intact(sampling_scheme):
    model, _ = _run(sampling_scheme, NEIGHBOR_FUTURE_SIMULATED)
    for mask in model.encoder.seen_a2a_masks:
        assert mask.all()


def test_modes_are_validated(sampling_scheme):
    _, synth, tok_map, tok_agent = _setup(sampling_scheme)
    with pytest.raises(ValueError, match="neighbor_future"):
        synth.sample_rollouts(
            tok_map, tok_agent, free_agent_indices=[0], n_samples=1, neighbor_future="nope"
        )
    assert set(NEIGHBOR_FUTURE_MODES) == {"logged", "hidden", "simulated"}


def test_edge_builder_is_restored(sampling_scheme):
    model, synth, tok_map, tok_agent = _setup(sampling_scheme)
    original = model.encoder.build_interaction_edge

    synth.sample_rollouts(
        tok_map, tok_agent, free_agent_indices=[0], n_samples=1,
        neighbor_future=NEIGHBOR_FUTURE_HIDDEN,
    )

    assert model.encoder.build_interaction_edge == original


def test_edge_builder_is_restored_after_an_exception(sampling_scheme):
    model, _, _, _ = _setup(sampling_scheme)
    original = model.encoder.build_interaction_edge
    hidden = torch.tensor([True, False, True])

    with pytest.raises(RuntimeError):
        with hide_agents_after_current_step(model.encoder, hidden):
            raise RuntimeError("boom")

    assert model.encoder.build_interaction_edge == original


def test_hiding_detects_a_changed_call_convention(sampling_scheme):
    model, _, _, _ = _setup(sampling_scheme)
    hidden = torch.tensor([True, False, True])
    with hide_agents_after_current_step(model.encoder, hidden):
        # first call is the history step and passes through untouched
        model.encoder.build_interaction_edge(
            pos_a=None, head_a=None, head_vector_a=None, batch_s=None,
            mask=torch.ones(N_AGENT, 2, dtype=torch.bool),
        )
        with pytest.raises(RuntimeError, match="mask"):
            model.encoder.build_interaction_edge(
                None, None, None, None, torch.ones(N_AGENT, 1, dtype=torch.bool)
            )


def test_hidden_mode_survives_rollout_batching(sampling_scheme):
    model = FakeSMART(n_agent=N_AGENT)
    synth = MotionSetSynthesizer(model, sampling_scheme, rollout_batch_size=3)
    tok_map, tok_agent = make_tokenized_map(), make_tokenized_agent(N_AGENT)

    synth.sample_rollouts(
        tok_map, tok_agent, free_agent_indices=[FREE_IDX], n_samples=3,
        neighbor_future=NEIGHBOR_FUTURE_HIDDEN,
    )

    # 3 replicated copies in one call: the hidden rows must line up per copy
    for mask in model.encoder.seen_a2a_masks[1:]:
        assert mask.shape[0] == N_AGENT * 3
        for copy in range(3):
            block = mask[copy * N_AGENT : (copy + 1) * N_AGENT]
            assert block[FREE_IDX].all()
            assert not block[0].any() and not block[2].any()

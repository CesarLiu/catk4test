import pytest
import torch

from tests.responsibility.conftest import SHIFT, FakeSMART

from src.responsibility.dataset_builder import (
    ResponsibilityObservation,
    causal_window_steps,
    group_sequences,
    last_usable_window_step,
    sequences_as_arrays,
    window_steps,
)
from src.responsibility.metrics import ResponsibilityConfig, ground_truth_boxes
from src.responsibility.motion_set import (
    MotionSetSynthesizer,
    isolated_agent_time_fields,
    shifted_context,
)

N_STEP = 91
NUM_HIST = 11


class _Store(dict):
    """Minimal stand-in for a PyG NodeStorage: item access over tensors."""


class _Data(dict):
    def __init__(self, n_agent=3, n_step=N_STEP):
        super().__init__()
        t = torch.arange(n_step).float()
        self["agent"] = _Store(
            position=torch.stack(
                [torch.stack([t + 10 * a, torch.zeros(n_step)], dim=-1) for a in range(n_agent)]
            ),
            heading=torch.zeros(n_agent, n_step),
            velocity=torch.ones(n_agent, n_step, 2),
            valid_mask=torch.ones(n_agent, n_step, dtype=torch.bool),
            shape=torch.tensor([[4.8, 2.0, 1.5]]).expand(n_agent, -1).clone(),
            type=torch.zeros(n_agent, dtype=torch.long),
            id=torch.arange(n_agent),
        )


def _cfg(**kw):
    return ResponsibilityConfig(num_historical_steps=NUM_HIST, **kw)


def _synth(sampling_scheme):
    return MotionSetSynthesizer(FakeSMART(n_agent=3), sampling_scheme)


def test_shifted_context_reanchors_the_current_step():
    data = _Data()
    original = data["agent"]["position"].clone()

    with shifted_context(data, k=30, num_historical_steps=NUM_HIST) as shifted:
        # index num_historical_steps-1 of the shifted view is absolute step 30
        assert shifted["agent"]["position"][0, NUM_HIST - 1, 0].item() == 30.0
        assert shifted["agent"]["position"].shape[1] == N_STEP - (30 - (NUM_HIST - 1))

    assert torch.equal(data["agent"]["position"], original), "context was not restored"


def test_shifted_context_future_slicing_lines_up():
    """The logged future of a shifted clip must start one step after k."""
    data = _Data()
    cfg = _cfg(metric_horizon=20)
    with shifted_context(data, k=40, num_historical_steps=NUM_HIST) as shifted:
        boxes = ground_truth_boxes(shifted, [0], NUM_HIST, horizon=20)
    assert boxes.pos[0, 0, 0].item() == 41.0
    assert boxes.pos[0, -1, 0].item() == 60.0


def test_shifted_context_rejects_impossible_steps():
    data = _Data()
    with pytest.raises(ValueError):
        with shifted_context(data, k=NUM_HIST - 2, num_historical_steps=NUM_HIST):
            pass


def test_shifted_context_restores_after_exception():
    data = _Data()
    original = data["agent"]["valid_mask"]
    with pytest.raises(RuntimeError):
        with shifted_context(data, k=20, num_historical_steps=NUM_HIST):
            raise RuntimeError("boom")
    assert data["agent"]["valid_mask"] is original


def test_isolated_agent_time_fields_protects_the_caller():
    data = _Data()
    original = data["agent"]["heading"]

    with isolated_agent_time_fields(data) as inner:
        inner["agent"]["heading"] += 1.0  # simulate in-place tokenizer edits
        assert inner["agent"]["heading"] is not original

    assert data["agent"]["heading"] is original
    assert torch.equal(original, torch.zeros_like(original)), "caller's tensor was mutated"


def test_window_steps_are_spaced_and_bounded(sampling_scheme):
    data = _Data()
    cfg = _cfg(metric_horizon=20, window_stride=5)
    steps = window_steps(data, cfg, _synth(sampling_scheme), stride=cfg.window_stride)

    assert steps[0] == NUM_HIST - 1
    assert all(b - a == 5 for a, b in zip(steps, steps[1:]))
    assert steps[-1] <= last_usable_window_step(N_STEP, cfg, SHIFT, rollout_horizon=20)
    assert len(steps) > 1, "a clip should yield a sequence, not a single observation"


def test_last_usable_window_step_leaves_room_for_the_rollout(sampling_scheme):
    cfg = _cfg(metric_horizon=20)
    k_max = last_usable_window_step(N_STEP, cfg, SHIFT, rollout_horizon=20)
    # a shifted clip at k_max must still carry enough token steps for the
    # decoder's step_current + horizon/shift lookups
    view_len = N_STEP - (k_max - (NUM_HIST - 1))
    token_columns = (view_len - 1) // SHIFT
    assert token_columns >= (NUM_HIST - 1) // SHIFT + 20 // SHIFT


def test_causal_window_steps_exclude_the_predicted_future(sampling_scheme):
    data = _Data()
    cfg = _cfg(metric_horizon=20, window_stride=5)
    predict_step = 50

    steps = causal_window_steps(data, cfg, _synth(sampling_scheme), predict_step, stride=5)

    assert len(steps) > 0
    for k in steps:
        # beta at k reads the log over [k+1, k+metric_horizon]; that window
        # must close at or before the step we predict from
        assert k + cfg.metric_horizon <= predict_step


def test_causal_window_steps_empty_when_prediction_is_too_early(sampling_scheme):
    data = _Data()
    cfg = _cfg(metric_horizon=20)
    steps = causal_window_steps(data, cfg, _synth(sampling_scheme), predict_step=15, stride=5)
    assert steps == []


def test_group_sequences_orders_by_context_step():
    obs = [
        ResponsibilityObservation("s", 7, 0, k=30, beta=torch.tensor([3.0, 0.0])),
        ResponsibilityObservation("s", 7, 0, k=10, beta=torch.tensor([1.0, 0.0])),
        ResponsibilityObservation("s", 7, 0, k=20, beta=torch.tensor([2.0, 0.0])),
        ResponsibilityObservation("s", 9, 1, k=10, beta=torch.tensor([9.0, 0.0])),
    ]
    grouped = group_sequences(obs)

    assert set(grouped) == {("s", 7), ("s", 9)}
    assert [o.k for o in grouped[("s", 7)]] == [10, 20, 30]

    sequences = sequences_as_arrays(grouped)
    lengths = sorted(len(s) for s in sequences)
    assert lengths == [1, 3]
    seq = next(s for s in sequences if len(s) == 3)
    assert list(seq[:, 0]) == [1.0, 2.0, 3.0]

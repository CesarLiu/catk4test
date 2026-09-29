"""Reward fine-tuning against a trainable stand-in policy: logits are a
learnable per-token bias, the sampler draws from the softmax, and a rollout's
x-position is its realized token, so tests can steer and read behaviour."""

import copy

import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from tests.responsibility.conftest import (
    AGENT_DECODER_STUB,
    FIRST_N_STEP,
    N_FUTURE_2HZ,
    N_TOKEN,
    SHIFT,
    FakeMapEncoder,
    FakeTokenProcessor,
)

from src.responsibility.dataset_builder import SCOPE_NEIGHBOURS, SCOPE_TOWARD_SDC, ResponsibilityObservation
from src.responsibility.finetune import (
    FinetuneConfig,
    PolicyRollout,
    RewardFinetuner,
    exact_kl,
    group_advantages,
    level_table,
    rollout_reward_features,
    save_finetuned_checkpoint,
)
from src.responsibility.geometry import BoxTrajectories, collision_cost, collision_cost_matrix
from src.responsibility.metrics import ResponsibilityConfig

NUM_HIST = 11
REWARDED_TOKEN = 3


def categorical_sampler(token_traj, token_traj_all, sampling_scheme, next_token_logits, *args, **kwargs):
    idx = Categorical(logits=next_token_logits.detach() / sampling_scheme.temp).sample()
    return idx, token_traj_all[torch.arange(idx.shape[0]), idx]


@pytest.fixture(autouse=True)
def sample_from_the_policy(monkeypatch):
    monkeypatch.setattr(AGENT_DECODER_STUB, "sample_next_token_traj", categorical_sampler)


class PolicyAgentEncoder(nn.Module):
    """Logits = a learnable bias per token (shared by agents and steps)."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(N_TOKEN))
        self.num_future_steps = N_FUTURE_2HZ * SHIFT
        self.shift = SHIFT

    def inference(self, tokenized_agent, map_feature, sampling_scheme):
        n_agent = tokenized_agent["valid_mask"].shape[0]
        n_steps = self.num_future_steps // self.shift
        pred_idx = tokenized_agent["gt_idx"].clone()
        pred_valid = tokenized_agent["valid_mask"].clone()
        logits = []
        for t in range(n_steps):
            n_step = FIRST_N_STEP + t
            step_logits = self.bias.expand(n_agent, N_TOKEN)
            idx, _ = AGENT_DECODER_STUB.sample_next_token_traj(
                token_traj=tokenized_agent["token_traj"],
                token_traj_all=tokenized_agent["token_traj_all"],
                sampling_scheme=sampling_scheme,
                next_token_logits=step_logits,
                pos_now=torch.zeros(n_agent, 2),
                head_now=torch.zeros(n_agent),
                pos_next_gt=tokenized_agent["gt_pos_raw"][:, n_step],
                head_next_gt=tokenized_agent["gt_head_raw"][:, n_step],
                valid_next_gt=tokenized_agent["gt_valid_raw"][:, n_step],
                token_agent_shape=tokenized_agent["token_agent_shape"],
            )
            pred_idx[:, n_step] = idx
            pred_valid[:, n_step] = pred_valid[:, n_step - 1]
            logits.append(step_logits)
        realized = pred_idx[:, FIRST_N_STEP : FIRST_N_STEP + n_steps].float()
        x = realized.repeat_interleave(self.shift, dim=1)
        return {
            "next_token_logits": torch.stack(logits, dim=1),
            "next_token_valid": pred_valid[:, 1:-1],
            "pred_idx": pred_idx,
            "pred_traj_10hz": torch.stack([x, torch.zeros_like(x)], dim=-1),
            "pred_head_10hz": torch.zeros_like(x),
        }


class PolicyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.map_encoder = FakeMapEncoder()
        self.agent_encoder = PolicyAgentEncoder()


class PolicySMART(nn.Module):
    def __init__(self, n_agent=3):
        super().__init__()
        self.encoder = PolicyDecoder()
        self.token_processor = FakeTokenProcessor(n_agent)


def _data(n_agent=3, n_step=91):
    pos = torch.zeros(n_agent, n_step, 3)
    pos[:, :, 1] = torch.arange(n_agent).float()[:, None] * 50.0  # far apart laterally
    return {"agent": {
        "position": pos,
        "heading": torch.zeros(n_agent, n_step),
        "velocity": torch.zeros(n_agent, n_step, 2),
        "valid_mask": torch.ones(n_agent, n_step, dtype=torch.bool),
        "shape": torch.tensor([[4.8, 2.0, 1.5]]).expand(n_agent, -1).clone(),
    }}


def _token_reward(rollout, data, agents, levels, ref_lp):
    idx = torch.as_tensor(agents)
    return (rollout.tokens[:, idx, 0] == REWARDED_TOKEN).float()


def _finetuner(model, kl_coef, n_rollouts=16, reward_fn=_token_reward):
    cfg = FinetuneConfig(n_rollouts=n_rollouts, horizon=10, kl_coef=kl_coef)
    return RewardFinetuner(model, reward_fn, cfg, ResponsibilityConfig(num_historical_steps=NUM_HIST))


def test_kl_to_the_reference_is_zero_before_training():
    torch.manual_seed(0)
    model = PolicySMART()
    with torch.no_grad():
        model.encoder.agent_encoder.bias.normal_()
    tuner = _finetuner(model, kl_coef=1.0)
    _, stats = tuner.loss(_data(), {0: 1})
    assert stats["kl"] == pytest.approx(0.0, abs=1e-6)


def test_reward_raises_the_probability_of_rewarded_behaviour():
    torch.manual_seed(0)
    model = PolicySMART()
    tuner = _finetuner(model, kl_coef=0.01)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    p0 = torch.softmax(model.encoder.agent_encoder.bias, -1)[REWARDED_TOKEN].item()
    for _ in range(40):
        loss, _ = tuner.loss(_data(), {0: 1, 2: 0})
        opt.zero_grad()
        loss.backward()
        opt.step()
    p1 = torch.softmax(model.encoder.agent_encoder.bias, -1)[REWARDED_TOKEN].item()
    assert p0 == pytest.approx(1 / N_TOKEN)
    assert p1 > 0.3


def test_kl_term_holds_the_policy_at_the_reference():
    torch.manual_seed(0)
    model = PolicySMART()
    tuner = _finetuner(model, kl_coef=50.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(40):
        loss, _ = tuner.loss(_data(), {0: 1})
        opt.zero_grad()
        loss.backward()
        opt.step()
    p = torch.softmax(model.encoder.agent_encoder.bias, -1)[REWARDED_TOKEN].item()
    assert p < 0.1


def test_unrewarded_agents_only_contribute_kl():
    torch.manual_seed(0)
    model = PolicySMART()
    tuner = _finetuner(model, kl_coef=1.0)
    loss, stats = tuner.loss(_data(), {})
    assert stats["pg"] == 0.0 and "reward" not in stats
    loss.backward()  # the KL term alone still yields a (zero) gradient
    assert model.encoder.agent_encoder.bias.grad is not None


def test_evaluate_reports_diagnostics():
    torch.manual_seed(0)
    tuner = _finetuner(PolicySMART(), kl_coef=0.1)
    out = tuner.evaluate(_data(), {0: 1})
    assert out["kl"] == pytest.approx(0.0, abs=1e-6)
    assert {"ade", "min_ade", "reward", "collision_rate"} <= set(out)
    assert out["min_ade"] <= out["ade"]


def test_group_advantages():
    r = torch.tensor([[1.0, 2.0], [3.0, 2.0]])
    adv = group_advantages(r)
    assert torch.allclose(adv.mean(0), torch.zeros(2))
    assert torch.allclose(adv[:, 1], torch.zeros(2))  # identical rewards: no signal
    assert torch.allclose(group_advantages(r, normalize=False)[:, 0], torch.tensor([-1.0, 1.0]))


def test_exact_kl():
    p = torch.tensor([[0.0, 0.0]])
    assert exact_kl(p, p).item() == pytest.approx(0.0)
    q = torch.log(torch.tensor([[0.9, 0.1]]))
    expected = 0.5 * torch.log(torch.tensor(0.5 / 0.9)) + 0.5 * torch.log(torch.tensor(0.5 / 0.1))
    assert exact_kl(p, q).item() == pytest.approx(expected.item(), rel=1e-5)


def test_finetune_config_validates():
    with pytest.raises(ValueError):
        FinetuneConfig(n_rollouts=1)
    with pytest.raises(ValueError):
        FinetuneConfig(reward_transform="nope")


def _obs(agent_idx, k, level, scope=SCOPE_NEIGHBOURS, scenario="s"):
    return ResponsibilityObservation(scenario, 100 + agent_idx, agent_idx, k, torch.zeros(2), level, scope)


def test_level_table_prefers_the_native_window_and_own_neighbours():
    table = level_table([
        _obs(0, 20, 2), _obs(0, 10, 1),  # nearest window wins
        _obs(1, 10, 3, scope=SCOPE_TOWARD_SDC), _obs(1, 15, 0),  # own-neighbour scope wins
        _obs(2, 60, 4),  # too far from the rollout's context step
        _obs(3, 10, None),  # unlabeled
    ], target_k=10, max_gap=10)
    assert table == {"s": {0: 1, 1: 0}}


def _rollout(xs, ys):
    """1 copy, agents at constant positions (xs[i], ys[i]) over 10 steps."""
    n = len(xs)
    traj = torch.zeros(1, n, 10, 2)
    traj[0, :, :, 0] = torch.tensor(xs, dtype=torch.float)[:, None]
    traj[0, :, :, 1] = torch.tensor(ys, dtype=torch.float)[:, None]
    return PolicyRollout(
        logits=torch.zeros(1, n, 2, N_TOKEN), tokens=torch.zeros(1, n, 2, dtype=torch.long),
        valid=torch.ones(1, n, 2, dtype=torch.bool), traj=traj, head=torch.zeros(1, n, 10),
    )


def test_collision_feature_counts_other_rollouts_but_not_self():
    rcfg = ResponsibilityConfig(num_historical_steps=NUM_HIST)
    data = _data(3)
    lp = torch.zeros(1, 3)
    apart = rollout_reward_features(_rollout([0, 0, 0], [0, 50, 100]), data, [0, 1], lp, rcfg)
    assert torch.all(apart[..., 0] == 0)
    overlap = rollout_reward_features(_rollout([0, 1, 0], [0, 0, 100]), data, [0, 1, 2], lp, rcfg)
    assert overlap[0, 0, 0] > 0 and overlap[0, 1, 0] > 0 and overlap[0, 2, 0] == 0


def test_collision_matrix_sums_to_collision_cost():
    torch.manual_seed(0)
    def boxes(n):
        return BoxTrajectories(
            pos=torch.randn(n, 5, 2) * 3, heading=torch.randn(n, 5), valid=torch.ones(n, 5, dtype=torch.bool),
            length=torch.full((n,), 4.8), width=torch.full((n,), 2.0),
        )
    ego, other = boxes(3), boxes(4)
    assert torch.allclose(collision_cost_matrix(ego, other).sum(-1), collision_cost(ego, other))


def test_checkpoint_keeps_the_base_format(tmp_path):
    base = tmp_path / "base.ckpt"
    torch.save({"state_dict": {"w": torch.zeros(2)}, "hyper_parameters": {"model_config": {"a": 1}},
                "optimizer_states": [1], "epoch": 3}, base)
    model = nn.Linear(2, 1)
    out = tmp_path / "tuned.ckpt"
    save_finetuned_checkpoint(model, str(base), str(out), extra={"step": 5})
    state = torch.load(out, weights_only=False)
    assert set(state["state_dict"]) == {"weight", "bias"}
    assert state["hyper_parameters"] == {"model_config": {"a": 1}}
    assert "optimizer_states" not in state and state["responsibility_finetune"] == {"step": 5}

"""Test scaffolding for the SMART-dependent responsibility modules.

``src/responsibility/motion_set.py`` imports
``src.smart.modules.agent_decoder`` at module level so it can monkey-patch
that module's ``sample_next_token_traj``. The real module pulls in
torch_cluster/torch_geometric, and exercising it would need a trained
checkpoint, so we register a deterministic stub under that name *before*
any test imports ``motion_set``. Both ``src.smart`` and
``src.smart.modules`` are empty packages, so only the leaf needs stubbing.

The stub lets us unit-test the parts of the counterfactual machinery that
are ours rather than SMART's: the ground-truth token forcing, its per-step
bookkeeping, agent removal, rollout batching, and output shapes.
"""

import sys
import types

import pytest
import torch

N_TOKEN = 32
N_STEP_TOKEN = 18  # 2Hz steps in the cached WOMD format
N_FUTURE_2HZ = 16  # steps SMARTAgentDecoder.inference rolls out
FIRST_N_STEP = 2  # n_step at t=0 inside inference()
SHIFT = 5
FREE_TOKEN = N_TOKEN - 1  # what the stub sampler always returns; never a GT index


def _stub_sample_next_token_traj(
    token_traj,
    token_traj_all,
    sampling_scheme,
    next_token_logits,
    pos_now,
    head_now,
    pos_next_gt,
    head_next_gt,
    valid_next_gt,
    token_agent_shape,
):
    """Deterministic stand-in for the real CAT-K sampler: always picks
    ``FREE_TOKEN``, so any agent that ends up with a different token must
    have been overridden by our ground-truth forcing wrapper."""
    n_agent = next_token_logits.shape[0]
    next_token_idx = torch.full((n_agent,), FREE_TOKEN, dtype=torch.long)
    next_token_traj_all = token_traj_all[torch.arange(n_agent), next_token_idx]
    return next_token_idx, next_token_traj_all


def row_indexed_sampler(
    token_traj,
    token_traj_all,
    sampling_scheme,
    next_token_logits,
    pos_now,
    head_now,
    pos_next_gt,
    head_next_gt,
    valid_next_gt,
    token_agent_shape,
):
    """Alternative stub whose token depends on the agent's row, so tests can
    tell replicated copies of a scene apart."""
    n_agent = next_token_logits.shape[0]
    next_token_idx = torch.arange(n_agent) % N_TOKEN
    next_token_traj_all = token_traj_all[torch.arange(n_agent), next_token_idx]
    return next_token_idx, next_token_traj_all


def _install_agent_decoder_stub():
    # `import src.smart.modules.agent_decoder as m` short-circuits on the full
    # name in sys.modules and then walks the attribute chain, so the parent
    # package must be imported (it is an empty module, no heavy deps) and must
    # carry the stub as an attribute.
    import src.smart.modules as modules_pkg

    module = types.ModuleType("src.smart.modules.agent_decoder")
    module.sample_next_token_traj = _stub_sample_next_token_traj
    sys.modules["src.smart.modules.agent_decoder"] = module
    modules_pkg.agent_decoder = module
    return module


AGENT_DECODER_STUB = _install_agent_decoder_stub()


def make_tokenized_agent(n_agent: int = 4, n_graphs: int = 1) -> dict:
    """Minimal ``tokenized_agent`` with the fields our code reads.

    ``gt_idx[:, s] = s`` so a forced agent's realized token at a given
    rollout step is self-identifying (it must equal that step's n_step).
    """
    gt_idx = torch.arange(N_STEP_TOKEN).unsqueeze(0).expand(n_agent, -1).clone()
    return {
        "num_graphs": n_graphs,
        "batch": torch.zeros(n_agent, dtype=torch.long),
        "valid_mask": torch.ones(n_agent, N_STEP_TOKEN, dtype=torch.bool),
        "gt_idx": gt_idx,
        "gt_pos": torch.zeros(n_agent, N_STEP_TOKEN, 2),
        "gt_heading": torch.zeros(n_agent, N_STEP_TOKEN),
        "gt_pos_raw": torch.zeros(n_agent, N_STEP_TOKEN, 2),
        "gt_head_raw": torch.zeros(n_agent, N_STEP_TOKEN),
        "gt_valid_raw": torch.ones(n_agent, N_STEP_TOKEN, dtype=torch.bool),
        "gt_z_raw": torch.zeros(n_agent),
        "type": torch.zeros(n_agent, dtype=torch.long),
        "shape": torch.tensor([[4.8, 2.0, 1.5]]).expand(n_agent, -1).clone(),
        "ego_mask": torch.zeros(n_agent, dtype=torch.bool),
        "token_agent_shape": torch.tensor([[2.0, 4.8]]).expand(n_agent, -1).clone(),
        "token_traj": torch.zeros(n_agent, N_TOKEN, 4, 2),
        "token_traj_all": torch.arange(N_TOKEN)
        .view(1, N_TOKEN, 1, 1, 1)
        .expand(n_agent, N_TOKEN, 6, 4, 2)
        .float()
        .clone(),
        "trajectory_token_veh": torch.zeros(N_TOKEN, 8),
        "trajectory_token_ped": torch.zeros(N_TOKEN, 8),
        "trajectory_token_cyc": torch.zeros(N_TOKEN, 8),
    }


def make_tokenized_map(n_pl: int = 6, n_graphs: int = 1) -> dict:
    return {
        "position": torch.zeros(n_pl, 2),
        "orientation": torch.zeros(n_pl),
        "token_idx": torch.zeros(n_pl, dtype=torch.long),
        "type": torch.zeros(n_pl, dtype=torch.long),
        "pl_type": torch.zeros(n_pl, dtype=torch.long),
        "light_type": torch.zeros(n_pl, dtype=torch.long),
        "batch": torch.zeros(n_pl, dtype=torch.long),
        "token_traj_src": torch.zeros(N_TOKEN, 22),
    }


class FakeAgentEncoder:
    """Stands in for ``SMARTAgentDecoder``, reproducing the parts of its
    ``inference`` our wrapper depends on: one ``sample_next_token_traj``
    call per future step, looked up on the module (not captured at import
    time) so monkey-patching takes effect, and an output dict with the keys
    we consume."""

    def __init__(self, num_future_steps: int = N_FUTURE_2HZ * SHIFT):
        self.num_future_steps = num_future_steps
        self.shift = SHIFT
        self.seen_valid_masks = []  # one entry per inference() call
        self.seen_n_agent = []
        self.seen_a2a_masks = []  # one entry per rollout step

    def build_interaction_edge(self, pos_a, head_a, head_vector_a, batch_s, mask):
        """Same call convention as SMARTAgentDecoder's; we only record the
        agent-agent visibility mask so tests can see who was reachable."""
        self.seen_a2a_masks.append(mask.clone())
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 1)

    def inference(self, tokenized_agent, map_feature, sampling_scheme):
        n_agent = tokenized_agent["valid_mask"].shape[0]
        n_future_2hz = self.num_future_steps // self.shift
        # the real decoder offsets map and agent batches by the same graph
        # count; a mismatch here means the copies were replicated inconsistently
        n_map_graphs = int(map_feature["batch"].max()) + 1
        if n_map_graphs != int(tokenized_agent["num_graphs"]):
            raise AssertionError(
                f"map batch spans {n_map_graphs} graphs but agents claim {tokenized_agent['num_graphs']}"
            )
        self.seen_valid_masks.append(tokenized_agent["valid_mask"].clone())
        self.seen_n_agent.append(n_agent)
        step_current_2hz = 2

        pred_idx = tokenized_agent["gt_idx"].clone()
        pred_valid = tokenized_agent["valid_mask"].clone()
        logits = []
        for t in range(n_future_2hz):
            n_step = FIRST_N_STEP + t
            # mirrors the real decoder: the first call spans the history
            # steps, later ones only the newest step
            hist_step = step_current_2hz if t == 0 else 1
            self.build_interaction_edge(
                pos_a=torch.zeros(n_agent, hist_step, 2),
                head_a=torch.zeros(n_agent, hist_step),
                head_vector_a=torch.zeros(n_agent, hist_step, 2),
                batch_s=tokenized_agent["batch"].repeat(hist_step),
                mask=pred_valid[:, :hist_step],
            )
            next_token_logits = torch.zeros(n_agent, N_TOKEN)
            next_token_idx, _ = AGENT_DECODER_STUB.sample_next_token_traj(
                token_traj=tokenized_agent["token_traj"],
                token_traj_all=tokenized_agent["token_traj_all"],
                sampling_scheme=sampling_scheme,
                next_token_logits=next_token_logits,
                pos_now=torch.zeros(n_agent, 2),
                head_now=torch.zeros(n_agent),
                pos_next_gt=tokenized_agent["gt_pos_raw"][:, n_step],
                head_next_gt=tokenized_agent["gt_head_raw"][:, n_step],
                valid_next_gt=tokenized_agent["gt_valid_raw"][:, n_step],
                token_agent_shape=tokenized_agent["token_agent_shape"],
            )
            pred_idx[:, n_step] = next_token_idx
            logits.append(next_token_logits)

        # trajectory encodes the realized token so tests can read it back
        realized = pred_idx[:, FIRST_N_STEP : FIRST_N_STEP + n_future_2hz]  # [n_agent, H2]
        pred_traj = realized.float().unsqueeze(-1).repeat_interleave(self.shift, dim=1)
        pred_traj = torch.stack([pred_traj.squeeze(-1), torch.zeros_like(pred_traj.squeeze(-1))], dim=-1)

        return {
            "next_token_logits": torch.stack(logits, dim=1),  # [n_agent, H2, n_token]
            "next_token_valid": pred_valid[:, 1:-1],  # [n_agent, 16], as in SMART
            "pred_idx": pred_idx,  # [n_agent, 18]
            "pred_valid": pred_valid,
            "pred_traj_10hz": pred_traj,  # [n_agent, H2*shift, 2]
            "pred_head_10hz": torch.zeros(n_agent, n_future_2hz * self.shift),
            "pred_z_10hz": torch.zeros(n_agent, n_future_2hz * self.shift),
        }


class FakeMapEncoder:
    """Stands in for ``SMARTMapDecoder``; counts how often the map is encoded."""

    def __init__(self):
        self.calls = 0
        self.seen_n_pl = []

    def __call__(self, tokenized_map):
        self.calls += 1
        n_pl = tokenized_map["position"].shape[0]
        self.seen_n_pl.append(n_pl)
        return {
            "pt_token": torch.zeros(n_pl, 4),
            "position": tokenized_map["position"],
            "orientation": tokenized_map["orientation"],
            "batch": tokenized_map["batch"],
        }


class FakeDecoder:
    """Stands in for ``SMARTDecoder``: a map encoder plus an agent decoder.
    Attributes not defined here are forwarded to the agent decoder, which is
    where the rollout loop (and everything tests inspect) lives."""

    def __init__(self, num_future_steps: int = N_FUTURE_2HZ * SHIFT):
        self.map_encoder = FakeMapEncoder()
        self.agent_encoder = FakeAgentEncoder(num_future_steps=num_future_steps)

    def inference(self, tokenized_map, tokenized_agent, sampling_scheme):
        return self.agent_encoder.inference(tokenized_agent, self.map_encoder(tokenized_map), sampling_scheme)

    def __getattr__(self, name):
        return getattr(self.__dict__["agent_encoder"], name)


class FakeTokenProcessor:
    def __init__(self, n_agent: int = 4):
        self.n_agent = n_agent
        self.n_token_agent = N_TOKEN

    def __call__(self, data):
        n_agent = self.n_agent
        try:  # follow the scenario when handed real-shaped data
            n_agent = data["agent"]["position"].shape[0]
        except (KeyError, TypeError, AttributeError):
            pass
        return make_tokenized_map(), make_tokenized_agent(n_agent)


class FakeSMART:
    """Duck-typed stand-in for the ``SMART`` LightningModule."""

    def __init__(self, n_agent: int = 4, num_future_steps: int = N_FUTURE_2HZ * SHIFT):
        self.training = False
        self.encoder = FakeDecoder(num_future_steps=num_future_steps)
        self.token_processor = FakeTokenProcessor(n_agent)


@pytest.fixture
def fake_model():
    return FakeSMART()


@pytest.fixture
def sampling_scheme():
    from omegaconf import OmegaConf

    return OmegaConf.create({"criterium": "topk_prob", "num_k": 4, "temp": 1.0})

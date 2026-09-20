"""Counterfactual motion-set synthesis (paper Sec. III-A) on top of SMART.

SMART is a *jointly* autoregressive multi-agent model: at every rollout
step every agent's next token depends on every other agent's just-sampled
state. To realize the paper's open-loop counterfactual -- "what else
could agent a have done, given the same real scene context" -- we need to
roll out only the queried agent(s) while every other agent is teacher-
forced onto its logged ground-truth continuation. We do this without
touching model internals by temporarily monkey-patching the module-level
``sample_next_token_traj`` function that ``SMARTAgentDecoder.inference``
calls once per step (see ``src/smart/modules/agent_decoder.py``): our
wrapper calls the original sampler (so attention/features are computed
identically to a real rollout) and then overwrites the sampled token for
every non-free agent with its ground-truth token for that step.

To realize the "agent removed from the scene" counterfactual
(``x_k^{-a}`` in the paper), we instead zero out that agent's validity
mask so it is excluded from every attention edge (temporal, agent-agent,
map-agent) that SMART builds, exactly like the ``ego_only`` masking
already used by ``WOSACMetrics`` in this repo, but with the roles
reversed (mask one agent out instead of masking all-but-one).
"""

import contextlib
from typing import Dict, Optional, Sequence

import torch
from omegaconf import DictConfig
from torch import Tensor

import src.smart.modules.agent_decoder as agent_decoder_module


class _ForcedTokenSampler:
    """Callable replacement for ``sample_next_token_traj`` (see module docstring)."""

    def __init__(self, tokenized_agent: Dict[str, Tensor], free_mask: Tensor):
        self.tokenized_agent = tokenized_agent
        self.forced_mask = ~free_mask
        # SMARTAgentDecoder.inference calls the sampler once per t=0..15,
        # at n_step = 2, 3, ..., 17 (see agent_decoder.py::inference).
        self.step_counter = 2
        self.original = agent_decoder_module.sample_next_token_traj

    def __call__(
        self,
        token_traj: Tensor,
        token_traj_all: Tensor,
        sampling_scheme: DictConfig,
        next_token_logits: Tensor,
        pos_now: Tensor,
        head_now: Tensor,
        pos_next_gt: Tensor,
        head_next_gt: Tensor,
        valid_next_gt: Tensor,
        token_agent_shape: Tensor,
    ):
        next_token_idx, next_token_traj_all = self.original(
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
        )
        n_step = self.step_counter
        self.step_counter += 1
        if not self.forced_mask.any():
            return next_token_idx, next_token_traj_all

        gt_idx_here = self.tokenized_agent["gt_idx"][:, n_step]  # [n_agent]
        gt_valid_here = self.tokenized_agent["valid_mask"][:, n_step]  # [n_agent]
        force = self.forced_mask & gt_valid_here
        if force.any():
            range_a = torch.arange(
                next_token_traj_all.shape[0], device=next_token_traj_all.device
            )
            forced_traj = token_traj_all[range_a, gt_idx_here]  # [n_agent, 6, 4, 2]

            next_token_idx = next_token_idx.clone()
            next_token_idx[force] = gt_idx_here[force]
            next_token_traj_all = next_token_traj_all.clone()
            next_token_traj_all[force] = forced_traj[force]
        return next_token_idx, next_token_traj_all


@contextlib.contextmanager
def force_ground_truth_tokens(tokenized_agent: Dict[str, Tensor], free_mask: Tensor):
    """Teacher-forces every agent outside ``free_mask`` onto its GT token."""
    sampler = _ForcedTokenSampler(tokenized_agent, free_mask)
    agent_decoder_module.sample_next_token_traj = sampler
    try:
        yield
    finally:
        agent_decoder_module.sample_next_token_traj = sampler.original


def remove_agents(
    tokenized_agent: Dict[str, Tensor], agent_indices: Sequence[int]
) -> Dict[str, Tensor]:
    """Returns a shallow copy of ``tokenized_agent`` with the given agents'
    validity zeroed out at every step, excluding them from all attention
    edges SMART builds during rollout (the ``x_k^{-a}`` counterfactual)."""
    out = dict(tokenized_agent)
    valid_mask = tokenized_agent["valid_mask"].clone()
    valid_mask[list(agent_indices), :] = False
    out["valid_mask"] = valid_mask
    return out


def find_neighbors(
    position: Tensor,  # [n_agent, n_step, 2]
    valid_mask: Tensor,  # [n_agent, n_step]
    query_idx: int,
    at_step: int = 10,
    radius: float = 60.0,
) -> Tensor:  # [n_agent] bool, excludes query_idx itself
    """Agents within ``radius`` meters of the queried agent at ``at_step``
    (default: the last observed history step, 10 = num_historical_steps-1)."""
    dist = torch.linalg.norm(
        position[:, at_step] - position[query_idx, at_step], dim=-1
    )
    neighbor_mask = (
        (dist <= radius) & valid_mask[:, at_step] & valid_mask[query_idx, at_step]
    )
    neighbor_mask[query_idx] = False
    return neighbor_mask


class MotionSetSynthesizer:
    """Draws counterfactual motion sets from a (loaded, ``eval()``-mode) SMART model."""

    def __init__(self, model, default_sampling_scheme: DictConfig):
        if model.training:
            raise ValueError("SMART model must be in eval() mode for rollout sampling.")
        self.model = model
        self.default_sampling_scheme = default_sampling_scheme

    @torch.no_grad()
    def tokenize(self, data):
        return self.model.token_processor(data)

    @torch.no_grad()
    def sample_rollouts(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        free_agent_indices: Sequence[int],
        n_samples: int,
        remove_agent_indices: Optional[Sequence[int]] = None,
        sampling_scheme: Optional[DictConfig] = None,
        output_agent_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, Tensor]:
        """Rolls out ``n_samples`` i.i.d. counterfactual continuations for
        the agents in ``free_agent_indices``; every other (non-removed)
        agent is teacher-forced onto ground truth at every step, and every
        agent in ``remove_agent_indices`` is excluded from the scene
        entirely (Eq. (1a)'s U^(a)_k(x_k), and the x_k^{-a} variant used
        for courtesy responsibility).

        ``output_agent_indices`` selects whose rollouts are returned, in the
        given order; it defaults to ``free_agent_indices``. Passing an empty
        ``free_agent_indices`` with an explicit output agent scores that
        agent's logged trajectory under teacher forcing.

        Returns a dict of tensors indexed [n_samples, n_output_agent, ...]:
            pred_traj_10hz:    [n_samples, n_free_agent, 80, 2]
            pred_head_10hz:    [n_samples, n_free_agent, 80]
            next_token_logits: [n_samples, n_free_agent, 16, n_token]
            next_token_idx:    [n_samples, n_free_agent, 16], the token actually
                realized at each of the 16 future 2Hz steps (use with
                next_token_logits to recover the realized trajectory's
                model log-probability, see reward_model.trajectory_log_prob)
            next_token_valid:  [n_samples, n_free_agent, 16]
        """
        n_agent = tokenized_agent["valid_mask"].shape[0]
        device = tokenized_agent["valid_mask"].device
        free_mask = torch.zeros(n_agent, dtype=torch.bool, device=device)
        free_mask[list(free_agent_indices)] = True

        work_agent = tokenized_agent
        if remove_agent_indices:
            work_agent = remove_agents(tokenized_agent, remove_agent_indices)
            free_mask = free_mask.clone()
            free_mask[list(remove_agent_indices)] = False

        scheme = sampling_scheme if sampling_scheme is not None else self.default_sampling_scheme
        out_idx = torch.as_tensor(
            list(output_agent_indices if output_agent_indices is not None else free_agent_indices),
            dtype=torch.long,
            device=device,
        )

        # A fresh context (and thus a fresh _ForcedTokenSampler with its step
        # counter reset to n_step=2) is required for *every* rollout: the
        # sampler is stepped once per call inside .inference(), so reusing
        # one across multiple rollouts would walk its step counter past the
        # 18-step gt_idx/valid_mask arrays on the second and later rollouts.
        traj_list, head_list, logits_list, idx_list, valid_list = [], [], [], [], []
        for _ in range(n_samples):
            with force_ground_truth_tokens(work_agent, free_mask):
                pred = self.model.encoder.inference(tokenized_map, work_agent, scheme)
            n_future_2hz = pred["next_token_logits"].shape[1]
            traj_list.append(pred["pred_traj_10hz"].index_select(0, out_idx))
            head_list.append(pred["pred_head_10hz"].index_select(0, out_idx))
            logits_list.append(pred["next_token_logits"].index_select(0, out_idx))
            # realized tokens and validity, aligned with next_token_logits' steps
            idx_list.append(pred["pred_idx"].index_select(0, out_idx)[:, 2 : 2 + n_future_2hz])
            valid_list.append(pred["next_token_valid"].index_select(0, out_idx)[:, :n_future_2hz])

        return {
            "pred_traj_10hz": torch.stack(traj_list, dim=0),
            "pred_head_10hz": torch.stack(head_list, dim=0),
            "next_token_logits": torch.stack(logits_list, dim=0),
            "next_token_idx": torch.stack(idx_list, dim=0),
            "next_token_valid": torch.stack(valid_list, dim=0),
            "output_agent_indices": out_idx,
        }

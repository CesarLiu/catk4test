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


# Per-agent / per-polyline tensors are replicated when a scene is copied;
# these are shared vocabularies indexed by token, not by node, so they are not.
_SHARED_AGENT_KEYS = {"trajectory_token_veh", "trajectory_token_ped", "trajectory_token_cyc"}
_SHARED_MAP_KEYS = {"token_traj_src"}


def _replicate_store(
    store: Dict[str, Tensor], n_nodes: int, n_graphs: int, n_copies: int, shared: set
) -> Dict[str, Tensor]:
    out = {}
    for key, value in store.items():
        if key in shared or not torch.is_tensor(value):
            out[key] = value
        elif key == "batch":
            out[key] = torch.cat([value + i * n_graphs for i in range(n_copies)], dim=0)
        elif value.shape[0] == n_nodes:
            out[key] = value.repeat(n_copies, *([1] * (value.dim() - 1)))
        else:
            out[key] = value
    return out


def replicate_scene(
    tokenized_map: Dict[str, Tensor],
    tokenized_agent: Dict[str, Tensor],
    n_copies: int,
):
    """Stacks ``n_copies`` independent copies of one scene into a single PyG
    batch, so that N i.i.d. rollouts come out of one ``.inference()`` call.

    SMART keys every attention edge off the ``batch`` index, so the copies
    cannot see each other; replicating is therefore equivalent to running
    the rollout N times, but pays the (dominant) map and attention setup
    cost once. The memory cost is dominated by ``token_traj_all``
    ([n_agent, n_token, 6, 4, 2]), which is why the number of copies per
    call is capped by ``MotionSetSynthesizer.rollout_batch_size``.
    """
    if n_copies == 1:
        return tokenized_map, tokenized_agent

    n_graphs = int(tokenized_agent["num_graphs"])
    n_agent = tokenized_agent["valid_mask"].shape[0]
    n_pl = tokenized_map["position"].shape[0]

    map_out = _replicate_store(tokenized_map, n_pl, n_graphs, n_copies, _SHARED_MAP_KEYS)
    agent_out = _replicate_store(tokenized_agent, n_agent, n_graphs, n_copies, _SHARED_AGENT_KEYS)
    agent_out["num_graphs"] = n_graphs * n_copies
    return map_out, agent_out


AGENT_TIME_FIELDS = ("position", "heading", "velocity", "valid_mask")


def _agent_time_tensors(data) -> Dict[str, Tensor]:
    store = data["agent"]
    found = {}
    for field in AGENT_TIME_FIELDS:
        try:
            value = store[field]
        except (KeyError, AttributeError):
            continue
        if torch.is_tensor(value):
            found[field] = value
    return found


@contextlib.contextmanager
def isolated_agent_time_fields(data):
    """Swaps the agent time-series tensors for copies for the duration of
    the block, so callees that mutate them in place cannot touch the
    caller's data, then restores the originals."""
    originals = _agent_time_tensors(data)
    store = data["agent"]
    try:
        for field, value in originals.items():
            store[field] = value.clone()
        yield data
    finally:
        for field, value in originals.items():
            store[field] = value


@contextlib.contextmanager
def shifted_context(data, k: int, num_historical_steps: int):
    """Re-anchors a cached scenario so that step ``k`` is the current step.

    The agent time series are sliced to start ``num_historical_steps - 1``
    steps before k, so everything downstream (tokenization, the rollout's
    own notion of "now", and the logged-future slicing in metrics) sees k
    as the present without any of it needing to know about windowing. The
    slices are views; restoration is exact.
    """
    start = k - (num_historical_steps - 1)
    if start < 0:
        raise ValueError(
            f"context step {k} needs {num_historical_steps - 1} steps of history"
        )
    originals = _agent_time_tensors(data)
    store = data["agent"]
    try:
        for field, value in originals.items():
            store[field] = value[:, start:]
        yield data
    finally:
        for field, value in originals.items():
            store[field] = value


# How much of the other agents' future the queried agent is allowed to
# condition on while its counterfactual alternatives are sampled.
NEIGHBOR_FUTURE_LOGGED = "logged"
NEIGHBOR_FUTURE_HIDDEN = "hidden"
NEIGHBOR_FUTURE_SIMULATED = "simulated"
NEIGHBOR_FUTURE_MODES = (
    NEIGHBOR_FUTURE_LOGGED,
    NEIGHBOR_FUTURE_HIDDEN,
    NEIGHBOR_FUTURE_SIMULATED,
)


@contextlib.contextmanager
def hide_agents_after_current_step(encoder, hidden_mask: Tensor):
    """Cuts the agent-agent attention edges of ``hidden_mask`` agents for
    every rollout step after the first.

    ``SMARTAgentDecoder.inference`` rebuilds the interaction graph each
    step from the agents' *current* poses, so an agent teacher-forced onto
    its log leaks its realized future to everyone else. The first call
    covers the history steps and is left alone; from the second on, the
    hidden agents' rows are masked out, and since ``build_interaction_edge``
    drops both in- and out-edges of masked agents (``subgraph``), the
    queried agent stops receiving updates about them. It still carries
    whatever it encoded about them at the current step, which is exactly
    the paper's "decision based on the current state and the interaction
    history" (Sec. VI-A).
    """
    owner = _horizon_owner(encoder)
    original = owner.build_interaction_edge
    state = {"calls": 0}

    def wrapped(*args, **kwargs):
        if "mask" not in kwargs:
            raise RuntimeError(
                "build_interaction_edge is expected to be called with mask=...; "
                "its call convention changed and neighbor-future hiding would silently break"
            )
        step = state["calls"]
        state["calls"] += 1
        if step > 0:  # call 0 covers the history steps
            mask = kwargs["mask"].clone()
            mask[hidden_mask] = False
            kwargs = {**kwargs, "mask": mask}
        return original(*args, **kwargs)

    owner.build_interaction_edge = wrapped
    try:
        yield
    finally:
        try:
            del owner.build_interaction_edge
        except AttributeError:
            owner.build_interaction_edge = original


def _horizon_owner(encoder):
    """SMARTDecoder delegates the rollout loop to its agent encoder; the
    test double implements it directly."""
    return getattr(encoder, "agent_encoder", encoder)


@contextlib.contextmanager
def temporary_rollout_horizon(encoder, horizon: Optional[int]):
    """Temporarily shortens the autoregressive rollout to ``horizon`` steps
    (at 10Hz). Because the rollout is autoregressive, stopping early yields
    exactly the prefix of a full-length rollout, so this is a pure compute
    saving whenever only the first few seconds are scored."""
    owner = _horizon_owner(encoder)
    if horizon is None:
        yield
        return

    shift = owner.shift
    if horizon <= 0 or horizon % shift != 0:
        raise ValueError(f"rollout_horizon must be a positive multiple of {shift}, got {horizon}")

    original = owner.num_future_steps
    owner.num_future_steps = min(horizon, original)
    try:
        yield
    finally:
        owner.num_future_steps = original


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

    def __init__(
        self,
        model,
        default_sampling_scheme: DictConfig,
        rollout_batch_size: int = 8,
    ):
        if model.training:
            raise ValueError("SMART model must be in eval() mode for rollout sampling.")
        self.model = model
        self.default_sampling_scheme = default_sampling_scheme
        self.rollout_batch_size = max(1, rollout_batch_size)

    @torch.no_grad()
    def tokenize(self, data):
        # TokenProcessor rewrites headings and back-fills positions in place
        # (_clean_heading / _extrapolate_agent_to_prev_token_step). That is
        # harmless for a training loop that sees each batch once, but we
        # tokenize the same scenario repeatedly (once per context window,
        # once per queried agent), so we hand it copies and keep the
        # caller's logged tensors pristine for the ground-truth comparisons.
        with isolated_agent_time_fields(data):
            return self.model.token_processor(data)

    @property
    def token_shift(self) -> int:
        return _horizon_owner(self.model.encoder).shift

    def round_horizon(self, horizon: Optional[int]) -> Optional[int]:
        """Rounds a horizon (in 10Hz steps) up to the next whole token step,
        which is the granularity the rollout loop can stop at."""
        if horizon is None:
            return None
        shift = _horizon_owner(self.model.encoder).shift
        return max(shift, -(-int(horizon) // shift) * shift)

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
        rollout_horizon: Optional[int] = None,
        neighbor_future: str = NEIGHBOR_FUTURE_LOGGED,
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
        agent's logged trajectory under teacher forcing (which requires
        ``neighbor_future="logged"``).

        ``neighbor_future`` controls how much of the other agents' future
        the free agents may condition on:

        - ``"logged"``   they replay their logged tokens and stay visible,
          so the free agent effectively foresees the real scene evolution.
          Faithful to "b replays its log", but more informed than the
          paper's open-loop assumption; biases safety responsibility up.
        - ``"hidden"``   they replay their logged tokens but drop out of
          the interaction graph after the current step, so the free agent
          decides on the current state and interaction history alone
          (the paper's stated open-loop reading, Sec. VI-A).
        - ``"simulated"`` nothing is teacher-forced: every agent samples
          its own future. No privileged information and fully in
          distribution, but the neighbours' realized motion is then a
          sample rather than the log.

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
        if neighbor_future not in NEIGHBOR_FUTURE_MODES:
            raise ValueError(
                f"neighbor_future must be one of {NEIGHBOR_FUTURE_MODES}, got {neighbor_future!r}"
            )
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

        n_agent = tokenized_agent["valid_mask"].shape[0]
        n_out = out_idx.shape[0]
        chunks = []
        remaining = n_samples
        while remaining > 0:
            chunks.append(min(remaining, self.rollout_batch_size))
            remaining -= chunks[-1]

        collected = {k: [] for k in ("traj", "head", "logits", "idx", "valid")}
        # A fresh context (and thus a fresh _ForcedTokenSampler with its step
        # counter reset to n_step=2) is required for *every* inference call:
        # the sampler is stepped once per call inside .inference(), so reusing
        # one across calls would walk its step counter past the 18-step
        # gt_idx/valid_mask arrays on the second and later calls.
        for chunk in chunks:
            rep_map, rep_agent = replicate_scene(tokenized_map, work_agent, chunk)
            rep_free_mask = free_mask.repeat(chunk)
            with contextlib.ExitStack() as stack:
                stack.enter_context(temporary_rollout_horizon(self.model.encoder, rollout_horizon))
                if neighbor_future != NEIGHBOR_FUTURE_SIMULATED:
                    stack.enter_context(force_ground_truth_tokens(rep_agent, rep_free_mask))
                if neighbor_future == NEIGHBOR_FUTURE_HIDDEN:
                    stack.enter_context(
                        hide_agents_after_current_step(self.model.encoder, ~rep_free_mask)
                    )
                pred = self.model.encoder.inference(rep_map, rep_agent, scheme)

            n_future_2hz = pred["next_token_logits"].shape[1]
            # rows of copy c live at [c*n_agent, (c+1)*n_agent)
            copy_offsets = n_agent * torch.arange(chunk, device=device)
            gather_idx = (out_idx.unsqueeze(0) + copy_offsets.unsqueeze(1)).reshape(-1)

            def _take(tensor):
                selected = tensor.index_select(0, gather_idx)
                return selected.view(chunk, n_out, *tensor.shape[1:])

            collected["traj"].append(_take(pred["pred_traj_10hz"]))
            collected["head"].append(_take(pred["pred_head_10hz"]))
            collected["logits"].append(_take(pred["next_token_logits"]))
            # realized tokens and validity, aligned with next_token_logits' steps
            collected["idx"].append(_take(pred["pred_idx"])[..., 2 : 2 + n_future_2hz])
            collected["valid"].append(_take(pred["next_token_valid"])[..., :n_future_2hz])

        return {
            "pred_traj_10hz": torch.cat(collected["traj"], dim=0),
            "pred_head_10hz": torch.cat(collected["head"], dim=0),
            "next_token_logits": torch.cat(collected["logits"], dim=0),
            "next_token_idx": torch.cat(collected["idx"], dim=0),
            "next_token_valid": torch.cat(collected["valid"], dim=0),
            "output_agent_indices": out_idx,
        }

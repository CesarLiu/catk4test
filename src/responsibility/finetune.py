"""Fine-tunes SMART with the per-level responsibility rewards (paper Sec. IV-B).

The paper learns one reward r_z per responsibility level z by discriminating
logged trajectories of that level from the model's own samples (stage 3),
and says the rewards can either re-rank a predictor's outputs (what
``selection.py`` and ``evaluate_selection.py`` do) or fine-tune the
predictor itself. This module does the latter, as KL-regularised policy
optimisation of the tokenized policy:

    max_theta  E_{tau ~ pi_theta} [ sum_a r_{z_a}(tau_a) ]
               - beta * sum_{a,t} KL( pi_theta(. | x_t) || pi_ref(. | x_t) )

For every training scene:

1. **Rollouts.** K joint closed-loop rollouts of ``horizon`` steps are drawn
   from the current policy, every agent free (as in CAT-K's closed-loop
   fine-tuning), by sampling the full token distribution -- so the sampled
   tokens are on-policy and their log-probabilities are exact. The K copies
   run as one batch (``motion_set.replicate_agents``), and the rollout keeps
   the graph for the gradient.
2. **Rewards.** Every agent with a responsibility level (from the stage-2
   labels, at the clip's native current step) gets its own level's reward
   on its own rollout, with the reward features of stage 3: collision with
   the *other agents' rollouts in the same scene copy*, road deviation,
   progress, and the rollout's mean per-token log-probability under the
   frozen *reference* model (under the policy itself that feature would
   reward whatever the policy already prefers).
3. **Advantages.** Group-relative, per agent over its K rollouts:
   (r - mean) / std (as in GRPO), so no value network is needed and a
   scene's difficulty cancels out.
4. **Loss.** -adv * log pi_theta(own tokens) for the rewarded agents, plus
   ``kl_coef`` times the KL to the reference model at every valid agent-step.
   The KL is exact rather than a sample estimate: a tokenized policy
   exposes both full next-token distributions, and the reference model's are
   obtained by replaying the policy's tokens through it (token forcing, as
   in ``metrics._replay_without``).

Credit assignment is per agent: an agent's advantage only multiplies the
log-probabilities of its own tokens, although the others' rollouts affect
its collision feature. This is the usual independent-learner simplification
for multi-agent policy gradients.
"""

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor

from src.responsibility.dataset_builder import SCOPE_NEIGHBOURS, get_map_points
from src.responsibility.geometry import (
    BoxTrajectories,
    collision_cost,
    collision_cost_matrix,
    final_displacement_progress,
    road_deviation_cost,
)
from src.responsibility.metrics import ResponsibilityConfig, others_future_boxes
from src.responsibility.motion_set import (
    force_ground_truth_tokens,
    isolated_agent_time_fields,
    replicate_agents,
    replicate_map,
    temporary_rollout_horizon,
)
from src.responsibility.reward_model import RewardNetBundle, mean_token_log_prob

REWARD_LOGIT = "logit"  # log D / (1 - D) of the level's discriminator
REWARD_PROB = "prob"  # D itself, in (0, 1)
REWARD_TRANSFORMS = (REWARD_LOGIT, REWARD_PROB)

OTHERS_ROLLOUT = "rollout"  # the other agents' rollouts in the same scene copy
OTHERS_CONFIG = "config"  # ResponsibilityConfig.others_future, as in stage 3
OTHERS_MODES = (OTHERS_ROLLOUT, OTHERS_CONFIG)


@dataclass
class FinetuneConfig:
    # rollouts per scene; the group the advantages are normalised over
    n_rollouts: int = 8
    # 10Hz steps rolled out and scored; the reward nets were trained on
    # features over the metric horizon, so keep the two equal
    horizon: int = 20
    kl_coef: float = 0.05
    reward_transform: str = REWARD_LOGIT
    others: str = OTHERS_ROLLOUT
    normalize_advantage: bool = True
    adv_eps: float = 1e-6

    def __post_init__(self):
        if self.n_rollouts < 2:
            raise ValueError("n_rollouts must be >= 2: advantages are relative to the other rollouts")
        if self.reward_transform not in REWARD_TRANSFORMS:
            raise ValueError(f"reward_transform must be one of {REWARD_TRANSFORMS}")
        if self.others not in OTHERS_MODES:
            raise ValueError(f"others must be one of {OTHERS_MODES}")


# ---------------------------------------------------------------------------
# which level each agent has
# ---------------------------------------------------------------------------


def level_table(
    observations: Sequence,
    target_k: int,
    max_gap: Optional[int] = 10,
) -> Dict[str, Dict[int, int]]:
    """{scenario_id: {agent_idx: level}} from level-labeled observations:
    per agent, the observation whose window starts closest to ``target_k``
    (the context step fine-tuning rolls out from), preferring the agent's
    responsibility toward all its neighbours over the toward-the-SDC scope.
    Observations more than ``max_gap`` steps away describe other parts of
    the clip and are ignored."""
    best: Dict[Tuple[str, int], Tuple[Tuple[bool, int], int]] = {}
    for o in observations:
        if o.level is None:
            continue
        gap = abs(o.k - target_k)
        if max_gap is not None and gap > max_gap:
            continue
        rank = (getattr(o, "scope", SCOPE_NEIGHBOURS) != SCOPE_NEIGHBOURS, gap)
        key = (o.scenario_id, int(o.agent_idx))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, int(o.level))
    table: Dict[str, Dict[int, int]] = {}
    for (scenario_id, agent), (_, level) in best.items():
        table.setdefault(scenario_id, {})[agent] = level
    return table


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------


@dataclass
class PolicyRollout:
    """K joint rollouts of one scene, reshaped to [K, n_agent, ...]."""

    logits: Tensor  # [K, n, S, V], with grad when drawn from the trained policy
    tokens: Tensor  # [K, n, S] realized tokens
    valid: Tensor  # [K, n, S] bool
    traj: Tensor  # [K, n, H, 2] at 10Hz
    head: Tensor  # [K, n, H]

    @property
    def present(self) -> Tensor:  # [K, n]: in the scene at the current step
        return self.valid[:, :, 0]


def full_vocabulary_scheme(tokenized_agent: Dict[str, Tensor]):
    """Samples from the whole softmax at temperature 1, so the realized
    tokens are exact samples of the policy whose log-probabilities the
    gradient uses (a truncated top-k would make them off-policy)."""
    n_token = int(tokenized_agent["token_traj"].shape[1])
    return OmegaConf.create({"criterium": "topk_prob", "num_k": n_token, "temp": 1.0})


def _parameters(module) -> List[Tensor]:
    return list(module.parameters()) if hasattr(module, "parameters") else []


def _split(x: Tensor, n_copies: int) -> Tensor:
    return x.reshape(n_copies, x.shape[0] // n_copies, *x.shape[1:])


def policy_rollouts(
    model,
    tokenized_map: Dict[str, Tensor],
    tokenized_agent: Dict[str, Tensor],
    n_rollouts: int,
    horizon: int,
) -> PolicyRollout:
    """Draws ``n_rollouts`` joint rollouts, keeping the autograd graph of
    the policy's logits. The map is encoded once (with grad only if the
    map encoder is being trained) and its features replicated."""
    encoder = model.encoder
    n_graphs = int(tokenized_agent["num_graphs"])
    train_map = any(p.requires_grad for p in _parameters(encoder.map_encoder))
    with torch.set_grad_enabled(torch.is_grad_enabled() and train_map):
        map_feature = encoder.map_encoder(tokenized_map)
    rep_agent = replicate_agents(tokenized_agent, n_rollouts)
    rep_map = replicate_map(map_feature, n_graphs, n_rollouts)
    with temporary_rollout_horizon(encoder, horizon):
        pred = encoder.agent_encoder.inference(rep_agent, rep_map, full_vocabulary_scheme(tokenized_agent))
    n_steps = pred["next_token_logits"].shape[1]
    return PolicyRollout(
        logits=_split(pred["next_token_logits"], n_rollouts),
        tokens=_split(pred["pred_idx"][:, 2 : 2 + n_steps], n_rollouts),
        valid=_split(pred["next_token_valid"][:, :n_steps], n_rollouts),
        traj=_split(pred["pred_traj_10hz"], n_rollouts),
        head=_split(pred["pred_head_10hz"], n_rollouts),
    )


@torch.no_grad()
def reference_logits(
    ref_model,
    tokenized_map: Dict[str, Tensor],
    tokenized_agent: Dict[str, Tensor],
    rollout: PolicyRollout,
    horizon: int,
) -> Tensor:  # [K, n, S, V]
    """The reference model's next-token logits along the policy's rollouts:
    every agent is forced onto the tokens it realized, so the reference sees
    exactly the states the policy visited."""
    k, n, s = rollout.tokens.shape
    encoder = ref_model.encoder
    rep = dict(replicate_agents(tokenized_agent, k))
    gt_idx, valid = rep["gt_idx"].clone(), rep["valid_mask"].clone()
    gt_idx[:, 2 : 2 + s] = rollout.tokens.reshape(k * n, s)
    valid[:, 2 : 2 + s] = rollout.valid.reshape(k * n, s)
    rep["gt_idx"], rep["valid_mask"] = gt_idx, valid
    rep_map = replicate_map(encoder.map_encoder(tokenized_map), int(tokenized_agent["num_graphs"]), k)
    free = torch.zeros(k * n, dtype=torch.bool, device=valid.device)
    with temporary_rollout_horizon(encoder, horizon), force_ground_truth_tokens(rep, free):
        pred = encoder.agent_encoder.inference(rep, rep_map, full_vocabulary_scheme(tokenized_agent))
    replayed = _split(pred["pred_idx"][:, 2 : 2 + s], k)
    if not torch.equal(replayed[rollout.valid], rollout.tokens[rollout.valid]):
        raise RuntimeError("replaying the policy's tokens through the reference model diverged")
    return _split(pred["next_token_logits"], k)


def token_log_probs(logits: Tensor, tokens: Tensor) -> Tensor:  # [..., S, V], [..., S] -> [..., S]
    return F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def exact_kl(logits: Tensor, ref_logits: Tensor) -> Tensor:  # [..., V] x2 -> [...]
    """KL(pi || pi_ref) of each next-token distribution, over the whole
    vocabulary."""
    log_p = F.log_softmax(logits, dim=-1)
    log_q = F.log_softmax(ref_logits, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def group_advantages(rewards: Tensor, normalize: bool = True, eps: float = 1e-6) -> Tensor:
    """[K, A] rewards of A agents over K rollouts -> advantages relative to
    each agent's own mean (and, with ``normalize``, in units of its std)."""
    centred = rewards - rewards.mean(dim=0, keepdim=True)
    if not normalize:
        return centred
    return centred / (rewards.std(dim=0, keepdim=True, unbiased=False) + eps)


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------


def rollout_reward_features(
    rollout: PolicyRollout,
    data,
    agents: Sequence[int],
    ref_mean_log_prob: Tensor,  # [K, n]
    rcfg: ResponsibilityConfig,
    others: str = OTHERS_ROLLOUT,
    map_points: Optional[Tensor] = None,
) -> Tensor:  # [K, A, 4]
    """Stage-3 reward features (collision, road deviation, progress,
    log-probability) of each agent's rollout, in the order of
    ``reward_model.extract_trajectory_features``."""
    k, n, horizon = rollout.traj.shape[:3]
    device = rollout.traj.device
    idx = torch.as_tensor(list(agents), dtype=torch.long, device=device)
    n_a = idx.numel()
    shape = data["agent"]["shape"].to(device)
    present = rollout.present  # [K, n]
    pos = rollout.traj[:, idx]  # [K, A, H, 2]
    ego_valid = present[:, idx, None].expand(k, n_a, horizon)

    if others == OTHERS_ROLLOUT:
        c_coll = []
        for c in range(k):
            everyone = BoxTrajectories(
                pos=rollout.traj[c], heading=rollout.head[c],
                valid=present[c][:, None].expand(n, horizon),
                length=shape[:, 0], width=shape[:, 1],
            )
            ego = BoxTrajectories(
                pos=pos[c], heading=rollout.head[c, idx], valid=ego_valid[c],
                length=shape[idx, 0], width=shape[idx, 1],
            )
            matrix = collision_cost_matrix(ego, everyone, rcfg.n_collision_circles, rcfg.collision_margin)
            matrix[torch.arange(n_a, device=device), idx] = 0.0  # not against itself
            c_coll.append(matrix.sum(dim=-1))
        c_coll = torch.stack(c_coll)  # [K, A]
    elif others == OTHERS_CONFIG:
        cols = []
        for i, a in enumerate(idx.tolist()):
            ego = BoxTrajectories(
                pos=pos[:, i], heading=rollout.head[:, a], valid=ego_valid[:, i],
                length=shape[a, 0].expand(k), width=shape[a, 1].expand(k),
            )
            rest = [j for j in range(n) if j != a]
            other = others_future_boxes(data, rest, rcfg, horizon)
            cols.append(collision_cost(ego, other, rcfg.n_collision_circles, rcfg.collision_margin))
        c_coll = torch.stack(cols, dim=1)
    else:
        raise ValueError(f"others must be one of {OTHERS_MODES}")

    if map_points is not None and map_points.numel() > 0:
        c_dev = road_deviation_cost(pos, ego_valid, map_points.to(device), rcfg.drivable_half_width)
    else:
        c_dev = torch.zeros_like(c_coll)
    origin = data["agent"]["position"][:, rcfg.num_historical_steps - 1, :2].to(device)[idx]  # [A, 2]
    progress = final_displacement_progress(pos, ego_valid, origin.unsqueeze(0).expand(k, -1, -1))
    return torch.stack([c_coll, c_dev, progress, ref_mean_log_prob[:, idx]], dim=-1)


RewardFn = Callable[[PolicyRollout, object, List[int], Dict[int, int], Tensor], Tensor]


def level_reward_fn(
    reward_nets: Dict[int, RewardNetBundle],
    rcfg: ResponsibilityConfig,
    cfg: FinetuneConfig,
) -> RewardFn:
    """Each agent's rollouts scored by its own level's reward network (the
    nearest trained level if its own has none)."""
    available = sorted(reward_nets)

    def reward(rollout, data, agents, levels, ref_mean_log_prob):
        features = rollout_reward_features(
            rollout, data, agents, ref_mean_log_prob, rcfg, cfg.others, get_map_points(data)
        )  # [K, A, 4]
        columns = []
        for i, a in enumerate(agents):
            level = min(available, key=lambda z: abs(z - levels[a]))
            net = reward_nets[level]
            score = net.logit if cfg.reward_transform == REWARD_LOGIT else net.reward
            columns.append(score(features[:, i]))
        return torch.stack(columns, dim=1)  # [K, A]

    return reward


# ---------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------


def frozen_copy(model):
    ref = copy.deepcopy(model)
    for p in _parameters(ref):
        p.requires_grad_(False)
    if hasattr(ref, "eval"):
        ref.eval()
    return ref


class RewardFinetuner:
    """Computes the fine-tuning loss of one scene; the optimiser, data order
    and checkpointing live in ``scripts/responsibility/finetune_reward.py``.

    ``model`` is kept in eval mode (dropout off, and SMART only returns 10Hz
    trajectories in eval mode) while gradients flow; which parameters train
    is decided by their ``requires_grad``.
    """

    def __init__(self, model, reward_fn: RewardFn, cfg: FinetuneConfig, rcfg: ResponsibilityConfig, ref_model=None):
        self.model = model
        self.ref_model = ref_model if ref_model is not None else frozen_copy(model)
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.rcfg = rcfg

    def tokenize(self, data):
        with torch.no_grad(), isolated_agent_time_fields(data):
            return self.model.token_processor(data)

    def _rollouts(self, data):
        tok_map, tok_agent = self.tokenize(data)
        rollout = policy_rollouts(self.model, tok_map, tok_agent, self.cfg.n_rollouts, self.cfg.horizon)
        ref = reference_logits(self.ref_model, tok_map, tok_agent, rollout, self.cfg.horizon)
        return rollout, ref

    def _scored_agents(self, rollout: PolicyRollout, levels: Dict[int, int]) -> List[int]:
        n = rollout.tokens.shape[1]
        return [a for a in sorted(levels) if a < n and bool(rollout.present[0, a])]

    def loss(self, data, levels: Dict[int, int]) -> Tuple[Tensor, Dict[str, float]]:
        """Loss of one scene. ``levels``: {agent index: responsibility level}
        of the agents to reward; everyone else only enters the KL term."""
        rollout, ref_logits = self._rollouts(data)
        valid = rollout.valid.to(rollout.logits.dtype)
        kl = exact_kl(rollout.logits, ref_logits)  # [K, n, S]
        kl_mean = (kl * valid).sum() / valid.sum().clamp(min=1.0)
        stats = {"kl": float(kl_mean.detach())}

        agents = self._scored_agents(rollout, levels)
        pg = kl_mean.new_zeros(())
        if agents:
            with torch.no_grad():
                ref_lp = mean_token_log_prob(token_log_probs(ref_logits, rollout.tokens), rollout.valid)
                rewards = self.reward_fn(rollout, data, agents, levels, ref_lp)  # [K, A]
                adv = group_advantages(rewards, self.cfg.normalize_advantage, self.cfg.adv_eps)
            idx = torch.as_tensor(agents, dtype=torch.long, device=rollout.logits.device)
            own_log_prob = (token_log_probs(rollout.logits[:, idx], rollout.tokens[:, idx]) * valid[:, idx]).sum(-1)
            pg = -(adv.to(own_log_prob.dtype) * own_log_prob).mean()
            stats.update(
                reward=float(rewards.mean()),
                reward_std=float(rewards.std(dim=0, unbiased=False).mean()),
                n_agents=float(len(agents)),
            )
        loss = pg + self.cfg.kl_coef * kl_mean
        stats.update(loss=float(loss.detach()), pg=float(pg.detach()))
        return loss, stats

    @torch.no_grad()
    def evaluate(self, data, levels: Dict[int, int]) -> Dict[str, float]:
        """Held-out diagnostics of one scene: reward of the rewarded agents,
        KL to the reference, and displacement of every present agent's
        rollouts from its logged future (mean over rollouts and best-of-K)."""
        rollout, ref_logits = self._rollouts(data)
        valid = rollout.valid.to(rollout.logits.dtype)
        kl = exact_kl(rollout.logits, ref_logits)
        out = {"kl": float((kl * valid).sum() / valid.sum().clamp(min=1.0))}

        horizon = rollout.traj.shape[2]
        agent = data["agent"]
        now = self.rcfg.num_historical_steps
        fut = slice(now, now + horizon)
        gt = agent["position"][:, fut, :2].to(rollout.traj.device)  # [n, H', 2]
        gt_valid = agent["valid_mask"][:, fut].to(rollout.traj.device)
        h = gt.shape[1]
        err = torch.linalg.norm(rollout.traj[:, :, :h] - gt.unsqueeze(0), dim=-1)  # [K, n, H']
        w = gt_valid.unsqueeze(0).to(err.dtype)
        ade = (err * w).sum(-1) / w.sum(-1).clamp(min=1.0)  # [K, n]
        scored = rollout.present[0] & gt_valid.any(-1)
        if bool(scored.any()):
            out["ade"] = float(ade[:, scored].mean())
            out["min_ade"] = float(ade[:, scored].min(dim=0).values.mean())

        agents = self._scored_agents(rollout, levels)
        if agents:
            ref_lp = mean_token_log_prob(token_log_probs(ref_logits, rollout.tokens), rollout.valid)
            rewards = self.reward_fn(rollout, data, agents, levels, ref_lp)
            out["reward"] = float(rewards.mean())
            features = rollout_reward_features(rollout, data, agents, ref_lp, self.rcfg, self.cfg.others)
            out["collision_rate"] = float((features[..., 0] > 0).float().mean())
        return out


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def save_finetuned_checkpoint(model, base_checkpoint: str, out_path: str, extra: Optional[Dict] = None) -> None:
    """Writes the fine-tuned weights in the base checkpoint's (Lightning)
    format, so ``load_smart_model`` and every script taking ``--ckpt`` read
    it unchanged. Training state of the base run (optimizer, schedulers) is
    dropped: it belongs to other weights."""
    state = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    state["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    for key in ("optimizer_states", "lr_schedulers"):
        state.pop(key, None)
    if extra is not None:
        state["responsibility_finetune"] = extra
    torch.save(state, out_path)

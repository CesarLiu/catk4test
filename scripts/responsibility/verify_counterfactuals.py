"""Checks, against a real checkpoint and real scenarios, the assumptions the
counterfactual machinery makes about SMART's decoder -- the ones the
stubbed unit tests in tests/responsibility cannot see.

  forcing   teacher-forcing agents onto their logged tokens reproduces the
            tokenized log exactly (and the raw log up to token quantization)
  removal   zeroing an agent's validity mask predicts the same for everyone
            else as deleting that agent from the scenario outright
  batching  N replicated scene copies in one call equal N separate calls
  hidden    with neighbor_future=hidden, perturbing a neighbour's logged
            future leaves the queried agent's predictions unchanged, while
            with neighbor_future=logged it does not -- i.e. the leak the
            switch exists for is real, and "hidden" closes it

All rollouts use argmax decoding (num_k=1) so that runs are comparable.
Exits non-zero if any check fails.

Example:
    python -m scripts.responsibility.verify_counterfactuals \
        --ckpt ckpts/clsft_E9.ckpt --data-root data/womd_cache/validation
"""

import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.responsibility.motion_set import (  # noqa: E402
    NEIGHBOR_FUTURE_HIDDEN,
    NEIGHBOR_FUTURE_LOGGED,
    MotionSetSynthesizer,
)
from src.responsibility.pipeline import (  # noqa: E402
    compose_responsibility_config,
    default_sampling_scheme,
    load_scenario,
    load_smart_model,
    resolve_device,
    scenario_id_of,
)

LOGIT_TOL = 1e-3
POSITION_TOL = 1e-3  # metres


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True, help="Cached scenario .pkl directory.")
    p.add_argument("--model-config", default="smart")
    p.add_argument("--scenario-index", type=int, default=0)
    p.add_argument("--horizon", type=int, default=40, help="Rollout steps @10Hz (multiple of 5).")
    p.add_argument("--perturbation", type=float, default=3.0, help="Lateral shift (m) of the neighbour's future.")
    p.add_argument("--plan-batch-size", type=int, default=32, help="Scene copies per call for the plan check.")
    p.add_argument("--device", default="auto", help="cuda, cpu, or auto (cuda if available).")
    return p.parse_args()


@contextlib.contextmanager
def _agent_fields_replaced(data, **replacements):
    store = data["agent"]
    originals = {k: store[k] for k in replacements}
    try:
        for k, v in replacements.items():
            store[k] = v
        yield data
    finally:
        for k, v in originals.items():
            store[k] = v


@contextlib.contextmanager
def _agent_deleted(data, j: int):
    """Removes agent ``j`` from every per-agent tensor of the scenario."""
    store = data["agent"]
    n_agent = store["position"].shape[0]
    keep = torch.arange(n_agent, device=store["position"].device) != j
    originals = {}
    for key in list(store.keys()):
        value = store[key]
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == n_agent:
            originals[key] = value
    try:
        for key, value in originals.items():
            store[key] = value[keep]
        yield data
    finally:
        for key, value in originals.items():
            store[key] = value


def _pick_interacting_pair(data, at_step: int, horizon: int):
    """Closest vehicle pair that is logged for the whole horizon."""
    pos = data["agent"]["position"][..., :2]
    valid = data["agent"]["valid_mask"][:, at_step : at_step + horizon + 1].all(-1)
    vehicle = data["agent"]["type"] == 0
    best = None
    for a in torch.nonzero(vehicle & valid).flatten().tolist():
        dist = torch.linalg.norm(pos[:, at_step] - pos[a, at_step], dim=-1)
        dist[a] = float("inf")
        dist[~valid] = float("inf")
        j = int(dist.argmin())
        if torch.isfinite(dist[j]) and (best is None or dist[j] < best[2]):
            best = (a, j, float(dist[j]))
    if best is None:
        raise RuntimeError("no pair of agents logged over the whole horizon in this scenario")
    return best


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail):
        self.rows.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    @property
    def ok(self):
        return all(ok for _, ok, _ in self.rows)


def _rollout(synth, data, free, horizon, **kw):
    tok_map, tok_agent = synth.tokenize(data)
    return synth.sample_rollouts(
        tok_map, tok_agent, free_agent_indices=free, n_samples=kw.pop("n_samples", 1),
        rollout_horizon=horizon, **kw,
    ), tok_agent


def _max_abs(a, b):
    return float((a - b).abs().max())


def check_forcing(synth, data, report, num_hist, horizon, shift):
    out, tok_agent = _rollout(synth, data, free=[], horizon=horizon,
                              output_agent_indices=range(data["agent"]["position"].shape[0]))
    n_steps = horizon // shift
    token_valid = tok_agent["valid_mask"][:, : 2 + n_steps].all(-1)
    raw_valid = data["agent"]["valid_mask"][:, num_hist : num_hist + horizon].all(-1)
    agents = token_valid & raw_valid
    n = int(agents.sum())

    pred = out["pred_traj_10hz"][0]  # [n_agent, H, 2]
    endpoints = pred[:, shift - 1 :: shift]  # 2Hz poses the rollout lands on
    tokenized = tok_agent["gt_pos"][:, 2 : 2 + n_steps]
    err_tok = torch.linalg.norm(endpoints - tokenized, dim=-1)[agents]
    report.add(
        "forcing reproduces the tokenized log",
        bool(err_tok.max() < POSITION_TOL),
        f"max error {err_tok.max():.2e} m over {n} fully logged agents",
    )

    raw = data["agent"]["position"][:, num_hist : num_hist + horizon, :2]
    err_raw = torch.linalg.norm(pred - raw, dim=-1)[agents]
    print(
        f"    info: vs the raw log (token quantization floor): mean {err_raw.mean():.3f} m, "
        f"p95 {err_raw.flatten().quantile(0.95):.3f} m, max {err_raw.max():.3f} m"
    )


def check_removal(synth, data, report, a, j, horizon):
    masked, _ = _rollout(synth, data, free=[a], horizon=horizon, remove_agent_indices=[j])
    present, _ = _rollout(synth, data, free=[a], horizon=horizon)
    with _agent_deleted(data, j):
        deleted, _ = _rollout(synth, data, free=[a - int(a > j)], horizon=horizon)

    diff_removed = _max_abs(masked["next_token_logits"], deleted["next_token_logits"])
    diff_present = _max_abs(masked["next_token_logits"], present["next_token_logits"])
    report.add(
        "mask removal equals deletion",
        diff_removed < LOGIT_TOL and diff_present > LOGIT_TOL,
        f"|logits(masked) - logits(deleted)| = {diff_removed:.2e}; "
        f"vs agent present = {diff_present:.2e} (must be clearly > 0 for the check to mean anything)",
    )


def check_batching(synth_1, synth_n, data, report, a, horizon, n):
    single, _ = _rollout(synth_1, data, free=[a], horizon=horizon, n_samples=n)
    batched, _ = _rollout(synth_n, data, free=[a], horizon=horizon, n_samples=n)
    diff = _max_abs(single["next_token_logits"], batched["next_token_logits"])
    same_tokens = torch.equal(single["next_token_idx"], batched["next_token_idx"])
    report.add(
        f"{n} batched copies equal {n} separate calls",
        diff < LOGIT_TOL and same_tokens,
        f"max logit diff {diff:.2e}, identical tokens: {same_tokens}",
    )


def check_hidden(synth, data, report, a, j, horizon, num_hist, offset):
    shifted = data["agent"]["position"].clone()
    shifted[j, num_hist:, 1] += offset

    def run(mode, perturbed):
        context = _agent_fields_replaced(data, position=shifted) if perturbed else contextlib.nullcontext()
        with context:
            out, _ = _rollout(synth, data, free=[a], horizon=horizon,
                              neighbor_future=mode, output_agent_indices=[a, j])
        return out

    base_l, pert_l = run(NEIGHBOR_FUTURE_LOGGED, False), run(NEIGHBOR_FUTURE_LOGGED, True)
    neighbour_moved = not torch.equal(base_l["next_token_idx"][:, 1], pert_l["next_token_idx"][:, 1])
    step0 = _max_abs(base_l["next_token_logits"][:, 0, 0], pert_l["next_token_logits"][:, 0, 0])
    leak = _max_abs(base_l["next_token_logits"][:, 0], pert_l["next_token_logits"][:, 0])

    base_h, pert_h = run(NEIGHBOR_FUTURE_HIDDEN, False), run(NEIGHBOR_FUTURE_HIDDEN, True)
    closed = _max_abs(base_h["next_token_logits"][:, 0], pert_h["next_token_logits"][:, 0])

    report.add(
        "perturbing the neighbour's future changes its replayed tokens",
        neighbour_moved,
        f"shifted agent {j}'s future by {offset} m laterally",
    )
    report.add(
        "the future cannot reach the first decision",
        step0 < LOGIT_TOL,
        f"step-0 logit change {step0:.2e}",
    )
    report.add(
        "logged mode leaks the neighbour's future",
        leak > LOGIT_TOL,
        f"queried agent's logits moved by up to {leak:.2e}",
    )
    report.add(
        "hidden mode closes the leak",
        closed < LOGIT_TOL,
        f"queried agent's logits moved by up to {closed:.2e}",
    )


def check_replay(synth, data, report, a, j, horizon, n=4):
    """Replay (forcing an agent onto given tokens) is what the exact courtesy
    estimator stands on. Replaying stochastically sampled tokens in the same
    context must reproduce the sampling pass's logits exactly; replaying
    them with a neighbour removed must not."""
    from src.responsibility.metrics import exact_courtesy_from_logits
    from src.responsibility.motion_set import RolloutJob

    tok_map, tok_agent = synth.tokenize(data)
    common = dict(rollout_horizon=horizon, keep_logits=True)
    (free,) = synth.run_jobs(tok_map, tok_agent, [RolloutJob(free=(a,), n_samples=n)], **common)
    tokens = free["next_token_idx"][:, 0]
    (same,) = synth.run_jobs(
        tok_map, tok_agent, [RolloutJob(free=(), n_samples=n, outputs=(a,), replay={a: tokens})], **common
    )
    (without,) = synth.run_jobs(
        tok_map, tok_agent, [RolloutJob(free=(), n_samples=n, removed=(j,), outputs=(a,), replay={a: tokens})],
        **common,
    )
    valid = free["next_token_valid"][:, 0]
    replayed_tokens = torch.equal(same["next_token_idx"][:, 0], tokens)
    diff = _max_abs(free["next_token_logits"], same["next_token_logits"])
    kl_same = float(exact_courtesy_from_logits(free["next_token_logits"][:, 0], same["next_token_logits"][:, 0], valid))
    kl_without = float(exact_courtesy_from_logits(free["next_token_logits"][:, 0], without["next_token_logits"][:, 0], valid))
    distinct = len({tuple(t.tolist()) for t in tokens})
    report.add(
        "replay reproduces sampled rollouts",
        replayed_tokens and diff < LOGIT_TOL and kl_same < 1e-4,
        f"{n} sampled rollouts ({distinct} distinct): tokens replayed exactly: {replayed_tokens}, "
        f"max logit diff {diff:.2e}, KL {kl_same:.2e}",
    )
    report.add(
        "exact courtesy detects the neighbour",
        kl_without > 1e-3,
        f"KL(p(xi_{a} | x) || p(xi_{a} | x without {j})) = {kl_without:.4f} nats over {horizon / 10:.1f} s",
    )


def check_plan(synth, data, report, num_hist, n_queried=4):
    """The packed, shared window plan must give the same betas as drawing
    every agent's and every pair's rollouts separately (argmax decoding
    makes both deterministic, so they are directly comparable)."""
    import time

    from src.responsibility.dataset_builder import get_map_points
    from src.responsibility.metrics import (
        ResponsibilityConfig,
        courtesy_responsibility,
        responsibility_for_agents,
        safety_responsibility,
    )
    from src.responsibility.motion_set import find_neighbors

    # Saturation and the courtesy distance skip are off so that the betas
    # being compared are not trivially zero; argmax decoding makes every
    # sample of a job identical, so without that most betas would vanish
    # and the comparison would prove nothing.
    cfg = ResponsibilityConfig(num_historical_steps=num_hist, n_safety_samples=4, n_courtesy_samples=4,
                               d_sat=None, courtesy_skip_distance=None)
    at = num_hist - 1
    pos = data["agent"]["position"][:, at, :2]
    n_agent = pos.shape[0]
    valid_now = data["agent"]["valid_mask"][:, at]
    neighbors_all = {
        a: find_neighbors(data["agent"]["position"][..., :2], data["agent"]["valid_mask"], a,
                          at_step=at, radius=cfg.neighbor_radius).nonzero(as_tuple=True)[0].tolist()
        for a in range(n_agent)
        if bool(valid_now[a]) and int(data["agent"]["type"][a]) == 0
    }
    # the vehicles with the closest neighbours, each with its 3 nearest
    def nearest(a):
        return sorted(neighbors_all[a], key=lambda b: float(torch.linalg.norm(pos[b] - pos[a])))[:3]

    queried = sorted(
        (a for a in neighbors_all if neighbors_all[a]),
        key=lambda a: float(torch.linalg.norm(pos[nearest(a)[0]] - pos[a])),
    )[:n_queried]
    neighbors_of = {a: nearest(a) for a in queried}
    all_idx, map_points = list(range(n_agent)), get_map_points(data)
    tok_map, tok_agent = synth.tokenize(data)

    t0 = time.perf_counter()
    packed, plan = responsibility_for_agents(synth, data, tok_map, tok_agent, queried, neighbors_of,
                                             all_idx, cfg, map_points=map_points)
    t_packed = time.perf_counter() - t0

    t0 = time.perf_counter()
    separate = {}
    for a in queried:
        s = safety_responsibility(synth, data, tok_map, tok_agent, a, neighbors_of[a], cfg)
        c = courtesy_responsibility(synth, data, tok_map, tok_agent, a, neighbors_of[a], all_idx, cfg,
                                    map_points=map_points)
        separate[a] = torch.stack([s["value"], c["value"]])
    t_separate = time.perf_counter() - t0

    diff = max(float((packed[a]["beta"] - separate[a]).abs().max()) for a in queried)
    nonzero_safety = sum(int(separate[a][0] != 0) for a in queried)
    nonzero_courtesy = sum(int(separate[a][1] != 0) for a in queried)
    report.add(
        "packed window plan equals per-pair computation",
        diff < POSITION_TOL and nonzero_safety > 0,
        f"max |beta| difference {diff:.2e} over {len(queried)} agents "
        f"(non-zero safety: {nonzero_safety}, non-zero courtesy: {nonzero_courtesy}; "
        f"at least one non-zero beta is required for the comparison to mean anything)",
    )
    print(f"    info: {plan.n_rollouts} rollouts in {len(plan.jobs)} jobs; "
          f"packed {t_packed:.1f} s vs separate {t_separate:.1f} s "
          f"at rollout_batch_size={synth.rollout_batch_size}")


def main():
    args = parse_args()
    torch.manual_seed(0)
    hydra_cfg = compose_responsibility_config(overrides=[f"model={args.model_config}"])
    device = resolve_device(args.device)
    model = load_smart_model(hydra_cfg.model.model_config, args.ckpt, device=device)
    num_hist = hydra_cfg.model.model_config.decoder.num_historical_steps

    argmax = default_sampling_scheme(num_k=1, temp=1.0)
    synth = MotionSetSynthesizer(model, argmax, rollout_batch_size=1)
    shift = synth.token_shift
    horizon = synth.round_horizon(args.horizon)

    data = load_scenario(args.data_root, args.scenario_index, device)

    a, j, dist = _pick_interacting_pair(data, num_hist - 1, horizon)
    n_agent = data["agent"]["position"].shape[0]
    print(f"scenario {scenario_id_of(data)}: {n_agent} agents; "
          f"queried agent {a}, neighbour {j} ({dist:.1f} m apart); horizon {horizon} steps")

    report = Report()
    with torch.no_grad():
        print("forcing")
        check_forcing(synth, data, report, num_hist, horizon, shift)
        print("removal")
        check_removal(synth, data, report, a, j, horizon)
        print("batching")
        check_batching(synth, MotionSetSynthesizer(model, argmax, rollout_batch_size=3),
                       data, report, a, horizon, n=3)
        print("hidden")
        check_hidden(synth, data, report, a, j, horizon, num_hist, args.perturbation)
        print("replay")
        check_replay(MotionSetSynthesizer(model, default_sampling_scheme(), rollout_batch_size=4),
                     data, report, a, j, horizon)
        print("plan")
        check_plan(MotionSetSynthesizer(model, argmax, rollout_batch_size=args.plan_batch_size),
                   data, report, num_hist)

    print("\nALL CHECKS PASSED" if report.ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()

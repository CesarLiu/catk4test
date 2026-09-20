# Counterfactual-Responsibility Interpretability Framework

Implements Hsu, Leung, Chen, Fisac & Pavone, ["Interpretable Trajectory
Prediction for Autonomous Vehicles via Counterfactual
Responsibility"](https://saferobotics.princeton.edu/research/responsibility)
(IROS 2023) on top of this repository's SMART tokenized traffic model,
trained on the Waymo Open Motion Dataset (WOMD).

The paper's own reference implementation used BITS on nuScenes; this port
targets SMART/WOMD instead, since the method itself is model-agnostic (any
model producing a multimodal distribution over future motions qualifies).
All code lives under `src/responsibility/`, with CLI drivers under
`scripts/responsibility/`.

## What's implemented

| Paper section | Module | Notes |
|---|---|---|
| III-A, Motion set synthesis | `motion_set.py` | Counterfactual rollouts of one agent with every other agent teacher-forced to ground truth, or excluded from the scene entirely |
| III-B, Eq. 2-5, Safety & courtesy responsibility | `metrics.py` | CVaR-based safety responsibility; Boltzmann/KL courtesy responsibility (Sec. V-B's tractable approximation) |
| IV-A, HMM level abstraction | `hmm.py` | From-scratch Gaussian HMM (Baum-Welch EM), BIC model selection, Eq. 6 Bayes filter |
| IV-B / V-C, Reward-dependent prediction | `reward_model.py`, `dataset_builder.py`, `selection.py` | Per-level IRL-style discriminator reward nets; online argmax-reward trajectory selection |
| V-B, Ex-post analysis | `scripts/responsibility/ex_post_analysis.py` | Safety/courtesy scatter by level, BIC-vs-H curve |

## Key adaptation: counterfactual rollouts inside a jointly-autoregressive model

SMART samples every agent's next token jointly and autoregressively: at
each step, every agent's continuation depends on every other agent's
just-sampled state. The paper's counterfactual, by contrast, asks "what
else could agent *a* have done, holding the rest of the real scene fixed."

We reconcile these by temporarily monkey-patching the module-level
`sample_next_token_traj` function that `SMARTAgentDecoder.inference` calls
once per rollout step (`src/smart/modules/agent_decoder.py`). Our wrapper
calls the original sampler unchanged (so attention/features are computed
exactly like a real rollout) and then overwrites the sampled token, for
every agent outside the "free" (queried) set, with that agent's own
logged ground-truth token for that step. This is implemented in
`motion_set.force_ground_truth_tokens`. Excluding an agent from the scene
entirely (the paper's `x_k^{-a}`, used for courtesy responsibility)
instead zeroes that agent's validity mask so SMART's attention-edge
builders drop it from every edge, mirroring the `ego_only` masking already
used by this repo's `WOSACMetrics`.

## Scoping decisions and approximations (read before trusting numbers)

- **One responsibility observation per (scenario, agent) by default.**
  The paper computes a dense responsibility *sequence* per agent from
  nuScenes' 20s scenes at 0.1s intervals. This repo's cached WOMD scenarios
  are short (~9.1s), non-overlapping clips, so by default we compute one
  observation per queried agent at the clip's native current step. Fed to
  the HMM, this still gives a valid level abstraction (the emission model),
  just without a learned transition matrix beyond the uniform prior. See
  `dataset_builder.sliding_window_contexts` for an experimental way to get
  genuine within-clip sequences by re-slicing one cached scenario at
  several shifted "current steps."
- **CVaR convention.** We use the confidence-level convention where
  `alpha -> 1` means "average over the single worst sample" (an upper-tail
  average), because that is the limit the paper's own text states. The
  paper's in-text integral formula is written for the mirror-image
  (lower-tail) convention and does not itself reduce to that stated limit;
  see the docstring in `risk.py` for the full explanation.
- **Road-deviation cost is a distance-to-nearest-map-point proxy**, not a
  true signed distance to the drivable polygon (which this repo's cached
  WOMD map format does not reify). Use
  `scripts/responsibility/inspect_map_types.py` to find which
  `pt_token.pl_type` values are lanes/road edges in your cached data and
  pre-filter `map_points` for a tighter approximation.
- **Collision cost covers each vehicle with 3 circles** along its long
  axis (radius = half-diagonal of one longitudinal slice), not exact
  oriented-box overlap. A single inscribing circle is not usable here: a
  4.8x2.0 m vehicle would get r=2.4 m, so two cars in adjacent lanes
  (~3.5 m apart) would register a permanent collision, lighting up both
  the courtesy KL and the IRL collision feature during ordinary driving.
- **D_g saturates at `d_sat` (default 10 m).** The paper leaves it
  unsaturated, but reports a safety axis topping out near 1.2 m with
  almost all mass at zero, which an unsaturated min-distance difference
  cannot produce: against a distant neighbor any alternative motion
  "gains" metres for free. Saturating at an interaction range keeps the
  metric about interaction; pass `--d-sat 0` for the literal definition.
  `neighbor_radius` defaults to 30 m (the paper states none) for the same
  reason.
- **Responsibility is measured over a fixed `metric_horizon`** (default 20
  steps = 2 s, the paper's own prediction horizon) rather than SMART's
  full 8 s rollout, so beta stays comparable across context times.
- **Courtesy rollouts are skipped for pairs more than
  `courtesy_skip_distance` apart**, where the Boltzmann KL is provably
  zero (the two rewards differ only in the collision term against the
  queried agent). This is a compute shortcut with no effect on the
  non-zero values; set it to `0` to disable.
- **Vehicle-vehicle only by default** (`agent_type_filter=0`), matching the
  paper's stated scope; pass `--agent-type -1` to include all types (the
  underlying dynamics/geometry code is not vehicle-specific).

## Compute

Responsibility is expensive: per queried agent it is `n_safety_samples`
rollouts plus `2 * n_courtesy_samples` per neighbour. Three things keep it
tractable, all on by default:

- **Batched rollouts.** `MotionSetSynthesizer` stacks `rollout_batch_size`
  copies of the scene into one PyG batch, so N i.i.d. rollouts come out of
  one `.inference()` call (SMART keys every attention edge off the batch
  index, so copies cannot see each other). Raise `--rollout-batch-size`
  for speed, lower it if the GPU runs out of memory -- the cost is
  dominated by `token_traj_all`, roughly 12 MB per scene copy for 32
  agents.
- **Truncated rollouts.** Only `metric_horizon` steps are ever scored, and
  an autoregressive rollout stopped early is exactly the prefix of a full
  one, so the metric paths roll out 2 s instead of 8 s.
- **Skipped courtesy pairs.** Pairs further apart than
  `courtesy_skip_distance` have a provably zero Boltzmann KL and are
  skipped before any rollout.

Start with `--max-scenarios` on a small slice and check that the
safety/courtesy distributions look like the paper's Fig. 2 (most mass near
the origin) before launching a full sweep.

## Pipeline

```bash
# 1. Compute responsibility observations over a cached WOMD split
python -m scripts.responsibility.compute_responsibility \
    --ckpt /path/to/smart.ckpt \
    --data-root /path/to/cache/validation \
    --out logs/responsibility/observations.pkl \
    --max-scenarios 200

# 2. Fit the HMM level abstraction (BIC sweep over H in {3,5,7,9,11})
python -m scripts.responsibility.fit_hmm \
    --observations logs/responsibility/observations.pkl \
    --out-hmm logs/responsibility/hmm.pkl \
    --out-labeled logs/responsibility/observations_labeled.pkl \
    --out-summary logs/responsibility/hmm_summary.json

# 3. Train per-level reward networks
python -m scripts.responsibility.train_reward \
    --ckpt /path/to/smart.ckpt \
    --data-root /path/to/cache/validation \
    --labeled-observations logs/responsibility/observations_labeled.pkl \
    --out logs/responsibility/reward_nets.pt

# 4. Online responsibility-aware prediction demo
python -m scripts.responsibility.predict_demo \
    --ckpt /path/to/smart.ckpt \
    --data-root /path/to/cache/validation \
    --hmm logs/responsibility/hmm.pkl \
    --reward-nets logs/responsibility/reward_nets.pt \
    --scenario-index 0

# Ex-post analysis plots (paper Fig. 2-3 style)
python -m scripts.responsibility.ex_post_analysis \
    --labeled-observations logs/responsibility/observations_labeled.pkl \
    --bic-summary logs/responsibility/hmm_summary.json \
    --out-dir logs/responsibility/plots
```

All scripts run from the repo root (`catk/`) and load the SMART model
config via a small dedicated Hydra tree, `configs/responsibility/eval.yaml`
(see `src/responsibility/pipeline.compose_responsibility_config`), rather
than the full `src/run.py` training entry point.

## Testing

`tests/responsibility/` covers the model-independent math (geometry costs,
CVaR, the Gaussian HMM, and reward-network training) with synthetic data
and runs without a checkpoint or `torch_geometric`/`torch_scatter`/
`torch_cluster`:

```bash
python -m pytest tests/responsibility -q
```

The SMART-dependent modules (`motion_set.py`, and everything built on it)
require this repo's full environment (a loaded checkpoint and cached WOMD
data) to exercise end-to-end; they were verified by careful code review and
by unit-testing every piece of math they call, but not run against a real
model in this change.

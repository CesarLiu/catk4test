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

- **Sequences come from sweeping the current step within a clip.** The
  paper evaluates responsibility every 0.1 s across nuScenes' 20 s scenes;
  this repo's cached WOMD scenarios are short (~9.1 s), non-overlapping
  clips. We therefore sweep the *current step* inside each clip
  (`window_stride`, default 5 steps = 0.5 s), re-anchoring it with
  `motion_set.shifted_context`, so each agent still yields a genuine beta
  sequence and the HMM's transition matrix is fitted on real transitions
  rather than staying at its uniform prior. `--no-sweep-windows` falls
  back to one observation per clip. Windows are capped so that a shifted
  clip still has enough tokenized future for the decoder to roll out
  (`last_usable_window_step`).
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
- **How much future the queried agent sees is a switch**
  (`neighbor_future`, default `logged`). SMART rebuilds its interaction
  graph from the agents' current poses at every rollout step, so a
  neighbour teacher-forced onto its log leaks its *realized* future to
  the agent whose alternatives we are sampling. That is more information
  than the paper's open-loop assumption ("their controls do not change
  even after observing other agents' future states", Sec. VI-A) and it
  biases safety responsibility upward: forewarned alternatives avoid the
  neighbour more easily, inflating `D_g(xi) - D_g(xi_a)`, and worst
  exactly in the interactive scenes one cares about. The three modes are
  described under "Open-loop modes" below.
- **Levels used online are causal.** beta at context step k is computed
  from the logged future over [k+1, k+metric_horizon], so it is only known
  at k+metric_horizon. The online selection path therefore takes its
  responsibility history from `causal_window_steps`, which keeps only
  windows that have closed by the prediction step. Scoring a prediction
  with a level derived from the future it is being compared against would
  leak the answer; this is the paper's `z_{k-T_f}` indexing made explicit.
  The practical consequence is that the demo predicts from a later step
  (default 50) rather than the clip's native current step, which is too
  early for any window to have closed.
- **Vehicle-vehicle only by default** (`agent_type_filter=0`), matching the
  paper's stated scope; pass `--agent-type -1` to include all types (the
  underlying dynamics/geometry code is not vehicle-specific).

## Who is queried, and who counts as a neighbour

`--query` picks whose responsibility is measured: `vehicles` (default,
the paper's vehicle-vehicle scope), `sdc` (only the self-driving car,
WOMD `role[:, 0]`), `interest` (WOMD objects of interest / tracks to
predict) or `all`. With `--query sdc`, `--direction` chooses between the
SDC's responsibility toward the agents it interacts with (`outgoing`),
each of those agents' responsibility toward the SDC (`incoming`), or
both; the two are stored with different `scope`s and never mixed into one
HMM sequence.

Responsibility is a max over neighbours (Eq. 5), so the neighbour set
decides what it measures. `--neighbor-selection`:

| mode | a candidate is a neighbour if ... |
|---|---|
| `radius` | it is within `--neighbor-radius` now (the original rule) |
| `geometric` (default) | within `--prefilter-radius` now, and over the metric window its footprint gap is <= `--gap-threshold`, or its PET <= `--pet-threshold`, or its constant-velocity TTC <= `--ttc-threshold` |
| `influence` | the geometric rule, or its model influence on the queried agent >= `--influence-threshold` |

A plain radius misses fast crossing traffic that is still far away and
admits cars that never close in (their safety term saturates to 0 but
they still cost rollouts). The geometric evidence reads the same logged
window as beta itself, so it adds no information beyond what beta uses:

- **min gap**: closest approach of the two footprints (3-circle cover).
- **PET** (post-encroachment time): the smallest time difference with
  which the two occupy the same place, looking back over the observed
  history too, as long as one of the two occupations is in the future.
  For a car-following pair this is the time headway, which is why the
  default threshold is 2 s (the two-second rule); at 3 s a car cruising
  30 m behind at 10 m/s would count.
- **TTC**: first time the footprints touch under constant velocity. It
  uses the same circle cover as the gap: with bounding circles two cars
  side by side in adjacent lanes would already "collide".

**Model influence** of b on a is the exact KL, step by step along a's
logged trajectory with every agent replaying its log, between a's
next-token distribution with and without b in the scene -- the approach of
the paper's reference [23], and Eq. (4) evaluated along the log. It costs
one packed call of `1 + n_candidates` scene copies per window, because a
single "b removed" copy yields every queried agent's distributions.

Each observation keeps the evidence for every candidate considered
(`interaction`, with a `selected` flag) and the per-neighbour betas before
the max (`per_neighbour`), so thresholds can be recalibrated and the
critical neighbour identified after the fact.

## Open-loop modes

`--neighbor-future` selects what the queried agent conditions on while
its counterfactual alternatives are drawn. Only the *conditioning*
changes: safety responsibility is always measured against the neighbour's
logged trajectory, as Eq. (2)-(3) require.

| mode | neighbours do | queried agent sees | note |
|---|---|---|---|
| `logged` (default) | replay their logged tokens | their whole realized future | faithful to "b replays its log"; the agent is better informed than the paper assumes, which inflates beta_safety |
| `hidden` | replay their logged tokens | them only up to the current step | the paper's stated open-loop reading: decide on the current state and the interaction history |
| `simulated` | sample their own futures | a plausible simulated future | no privileged information and fully in distribution, but the neighbours' motion is a sample, not the log |

`hidden` is implemented by cutting the hidden agents' agent-agent
attention edges from the second rollout step onward
(`motion_set.hide_agents_after_current_step`); the first step spans the
history and is left intact, so the agent keeps what it encoded about its
neighbours at time k and simply stops receiving updates.

Running the same slice under two modes and differencing gives a reading
the paper does not have: the gap between `logged` and `hidden` is how
much of the apparent safety margin was actually supplied by the queried
agent being able to anticipate its neighbours, rather than by its own
caution.

```bash
for mode in logged hidden simulated; do
  python -m scripts.responsibility.compute_responsibility       --ckpt /path/to/smart.ckpt --data-root /path/to/cache/validation       --neighbor-future $mode --max-scenarios 50       --out logs/responsibility/observations_$mode.pkl
done
```

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

## Environment notes

- **Pin `torch_geometric==2.6.1`.** The README's unpinned
  `pip install torch_geometric` now resolves to 2.8, whose `BaseTransform`
  makes `forward` abstract; this repo's `WaymoTargetBuilder*` transforms
  override `__call__` instead, so the datamodule fails with *Can't
  instantiate abstract class WaymoTargetBuilderVal*. 2.6.1 works with the
  pinned torch 2.4.
- **`waymo-open-dataset` ships Linux wheels only.** On Windows, use WSL2
  (Ubuntu 22.04's Python 3.10 fits TF 2.12). Everything here runs on CPU,
  just slowly.
- **The WOMD testing split has no future ground truth**, so responsibility,
  which is measured against the logged future, can only be computed on
  training or validation.

## Verifying against a checkpoint

The unit tests stub SMART's sampler, so they cannot confirm that SMART
itself behaves as the counterfactual machinery assumes. Run this once per
checkpoint:

```bash
python -m scripts.responsibility.verify_counterfactuals     --ckpt ckpts/clsft_E9.ckpt --data-root data/womd_cache/validation
```

It checks, with argmax decoding so runs are comparable, that forcing
reproduces the tokenized log, that masking an agent out equals deleting it,
that batched scene copies equal separate calls, and that a perturbation of a
neighbour's logged future reaches the queried agent under `logged` but not
under `hidden`. On `clsft_E9` (scenario `101d7af08d9b56ae`, 36 agents):

| check | result |
|---|---|
| forcing vs tokenized log | 0.00 m (vs raw log: mean 0.034 m, p95 0.091 m, max 0.254 m -- the token quantization floor) |
| masked vs deleted | 6.7e-6 logit difference (1.61 with the agent kept) |
| 3 batched vs 3 separate | 8.6e-6, identical tokens |
| neighbour future perturbed by 3 m | `logged` moves the queried agent's logits by 0.454, `hidden` by 5.7e-6 |

The residual ~1e-5 differences are float32 accumulation-order noise
(`torch_scatter` sums in a thread-dependent order on CPU), five orders of
magnitude below the effects being tested.

## Testing

`tests/responsibility/` runs without a checkpoint, `torch_geometric`,
`torch_scatter` or `torch_cluster`. A stubbed agent decoder
(`conftest.py`) stands in for SMART's sampler, which lets the tests cover
not just the model-independent math (geometry costs, CVaR, the Gaussian
HMM, reward-network training) but also the counterfactual machinery
itself: ground-truth token forcing and its per-step bookkeeping, agent
removal, rollout batching and truncation, context shifting, causal window
selection, and an end-to-end pass through both responsibility metrics,
level assignment, reward features and online selection:

```bash
python -m pytest tests/responsibility -q
```

What the stub cannot check is whether the *model* behaves as assumed:
that forcing a neighbour onto its logged tokens really reproduces its
logged trajectory, that zeroing a validity mask really removes an agent
from every attention edge, and that replicated scene copies really stay
independent. Those hold by construction from reading SMART's decoder, but
confirming them needs a real checkpoint and cached WOMD data.

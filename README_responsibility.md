# Counterfactual responsibility on SMART / WOMD: server guide

How to install, verify and run the counterfactual-responsibility framework
(Hsu et al., IROS 2023, ported onto CAT-K's SMART) on a Linux GPU server,
and how to produce the demos. The method, its adaptations and every
design decision are documented in [docs/responsibility.md](docs/responsibility.md);
this file is the runbook.

Target: Ubuntu 24.04, NVIDIA H200 (any recent NVIDIA GPU works the same way).
All commands run from the repository root (`catk/`).

---

## 1. Install

Requirements: an NVIDIA driver that supports CUDA 12.1+ (`nvidia-smi`
works; R535 or newer), `git`, `curl`, ~20 GB of disk for the environment.

```bash
bash install/setup_server.sh
source ~/venvs/catk/bin/activate
```

The script creates a Python **3.11** virtualenv with [uv](https://docs.astral.sh/uv/)
(no root needed) and installs torch 2.4.1 (CUDA 12.1 build; runs on Hopper,
sm_90), `torch_geometric==2.6.1`, `torch_scatter`/`torch_cluster`, the
pinned requirements, and the Waymo reader. Why these pins:

- **Python 3.11, not the system 3.12**: `tensorflow-cpu==2.12` (WOMD reader)
  has no 3.12 wheels.
- **`torch_geometric==2.6.1`**: 2.8 makes `BaseTransform.forward` abstract
  and breaks this repo's data transforms.

Variants:

```bash
ENV_DIR=/scratch/venvs/catk bash install/setup_server.sh     # venv elsewhere
BACKEND=conda ENV_NAME=catk bash install/setup_server.sh     # use conda instead of uv
CUDA_TAG=cu124 bash install/setup_server.sh                  # other torch CUDA build
```

Check the environment (imports, GPU kernels, the 140+ unit tests, which
need neither data nor a checkpoint):

```bash
bash install/check_env.sh
```

## 2. Data

1. Accept the [WOMD license](https://waymo.com/open/terms) and download the
   **scenario** format (v1.2.x, `uncompressed/scenario/`). With `gsutil`
   after `gcloud auth login`:

   ```bash
   mkdir -p data/womd/validation data/womd/training
   gsutil -m cp "gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario/validation/*" data/womd/validation/
   # training is 1000 files (~1 TB); take a subset if you only need more clips:
   gsutil -m cp "gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario/training/training.tfrecord-0000*" data/womd/training/
   ```

   Only **training** and **validation** are usable: responsibility is
   measured against the logged future, and the testing split has none.

2. Cache the scenarios into the pickles SMART loads (one file per scenario):

   ```bash
   python -m src.data_preprocess --split validation --num_workers 32 \
       --input_dir data/womd --output_dir data/womd_cache
   ```

   This reads `data/womd/validation/*` and writes
   `data/womd_cache/validation/<scenario_id>.pkl` (~44k scenarios for the
   full validation split).

## 3. Checkpoint

Put a trained SMART checkpoint under `ckpts/` (CAT-K cannot distribute
WOMD-trained weights; train one with `scripts/train.sh`, see the main
[README](README.md), or use your own). The examples use
`ckpts/clsft_E9.ckpt` with the default `configs/model/smart.yaml`; pass
`--model-config <name>` for another model config.

Verify that the checkpoint behaves as the counterfactual machinery assumes
(forcing reproduces the log, masking equals deleting, batched copies are
independent, `hidden` blocks foresight), and repeat the environment check
with real data:

```bash
CKPT=ckpts/clsft_E9.ckpt DATA=data/womd_cache/validation bash install/check_env.sh
```

Expected values are in [docs/responsibility.md](docs/responsibility.md#verifying-against-a-checkpoint);
every check must print `PASS`, ending in `ALL CHECKS PASSED`.

## 4. Demo in one command

```bash
CKPT=ckpts/clsft_E9.ckpt DATA=data/womd_cache/validation bash scripts/responsibility/demo.sh
```

It runs every stage on a small slice (40 scenarios for fitting, 20 held-out
for evaluation) and writes to `logs/responsibility/demo/`:

| output | what it shows |
|---|---|
| `video/motion_sets.mp4` (and `.gif`, `frames/`) | the self-driving car's counterfactual motion set at every context step, coloured by model probability, against the logged futures of the car and of the agent it interacts with most; below, its safety responsibility towards that agent over time |
| `plots/` | safety-vs-courtesy scatter coloured by HMM level, BIC over the number of levels (paper Fig. 2-3) |
| `eval_summary.txt`, `eval.csv` | ADE/FDE of responsibility-aware candidate selection against ablations (see 6.) |
| `predict_demo.txt` | one online prediction: causal beta history, inferred level, selected vs most-likely candidate |

Scale it with `N_SCEN`, `N_EVAL`, `SHARDS`, or pass settings to every stage
with `EXTRA`, e.g. `EXTRA="--neighbor-future hidden"`. Re-running resumes.

Videos for other scenes (the queried agent defaults to the self-driving car,
the neighbour to the agent with the tightest logged interaction with it):

```bash
python -m scripts.responsibility.visualize_motion_sets \
    --ckpt ckpts/clsft_E9.ckpt --data-root data/womd_cache/validation \
    --scenario-index 7 --out-dir logs/responsibility/video_7 \
    --n-samples 32 --stride 5 --neighbor-future hidden
# --agent-index / --neighbor-index pick a specific pair; --view-radius fixes the zoom
```

## 5. Full runs

### 5.1 Pick the rollout batch size

Stage 1 is dominated by counterfactual rollouts, which are batched as scene
copies. Measure throughput and memory on this GPU:

```bash
python -m scripts.responsibility.benchmark_rollouts \
    --ckpt ckpts/clsft_E9.ckpt --data-root data/womd_cache/validation \
    --batch-sizes 64 128 256 512 1024
```

Take the batch size where copies/s stops improving, with memory headroom
for scenes busier than the benchmarked one (it picks a busy one by default),
and pass it as `--rollout-batch-size` (the default on GPU is 128).

### 5.2 Stage 1: responsibility observations (sharded, resumable)

```bash
scripts/responsibility/run_sharded.sh 8 logs/responsibility/run_sdc -- \
    --ckpt ckpts/clsft_E9.ckpt --data-root data/womd_cache/validation \
    --query sdc --direction both --courtesy-estimator exact \
    --neighbor-selection geometric --rollout-batch-size 256
```

- 8 processes share the visible GPUs round-robin (several per GPU keep it
  busy while others tokenize); logs go to `OUT_DIR/logs/shard_*.log`, one
  result file per scenario to `OUT_DIR/obs/`.
- **Resumable:** re-running the same command skips finished scenarios. Run
  it inside `tmux`/`nohup`.
- `OUT_DIR/config.json` records the settings; adding results computed
  with different settings is refused (use a new directory).
- To try a configuration first, add `--max-scenarios 20` (per shard).

The choices that matter most (all documented in docs/responsibility.md):

| flag | default | meaning |
|---|---|---|
| `--query` | `vehicles` | whose responsibility: `sdc`, `interest` (WOMD tracks to predict), `vehicles`, `all` |
| `--direction` | `outgoing` | with `sdc`: SDC towards others, `incoming`, or `both` |
| `--neighbor-selection` | `geometric` | neighbours = agents with real interaction evidence (footprint gap, post-encroachment time, time-to-collision); `influence` adds the model's own influence test; `radius` is the paper's plain radius |
| `--neighbor-future` | `logged` | what the queried agent may foresee of the others: `logged`, `hidden` (paper's open-loop reading), `simulated` |
| `--courtesy-estimator` | `boltzmann` | paper's Eq. 7 proxy, or `exact` model KL (usually more informative) |
| `--plausible-fraction` | `1.0` | take the safety CVaR over only the most probable fraction of samples |
| `--n-safety-samples`, `--n-courtesy-samples` | 40, 20 | Monte-Carlo sizes |
| `--window-stride` | 5 | context steps between responsibility windows in a clip |

### 5.3 Stage 2: responsibility levels (HMM)

```bash
python -m scripts.responsibility.fit_hmm \
    --observations logs/responsibility/run_sdc \
    --out-hmm logs/responsibility/run_sdc/hmm.pkl \
    --out-labeled logs/responsibility/run_sdc/labeled.pkl \
    --out-summary logs/responsibility/run_sdc/hmm_summary.json
python -m scripts.responsibility.ex_post_analysis \
    --labeled-observations logs/responsibility/run_sdc/labeled.pkl \
    --bic-summary logs/responsibility/run_sdc/hmm_summary.json \
    --out-dir logs/responsibility/run_sdc/plots
```

CPU only, minutes. `--n-states-grid` sets the BIC sweep (default 3 5 7 9 11).

### 5.4 Stage 3: reward networks

Pass the **same responsibility flags as stage 1**:

```bash
R="--query sdc --direction both --courtesy-estimator exact --neighbor-selection geometric"
python -m scripts.responsibility.train_reward --ckpt ckpts/clsft_E9.ckpt \
    --data-root data/womd_cache/validation $R \
    --labeled-observations logs/responsibility/run_sdc/labeled.pkl \
    --out logs/responsibility/run_sdc/reward_nets.pt
# the H = 1 ablation: one reward for all levels
python -m scripts.responsibility.train_reward --ckpt ckpts/clsft_E9.ckpt \
    --data-root data/womd_cache/validation $R --pool-levels \
    --labeled-observations logs/responsibility/run_sdc/labeled.pkl \
    --out logs/responsibility/run_sdc/reward_pooled.pt
```

## 6. Evaluation

Fit stages 1-3 on one set of scenarios (e.g. training files) and evaluate
on another (validation), or split one directory with `--first-scenario`:

```bash
python -m scripts.responsibility.evaluate_selection --ckpt ckpts/clsft_E9.ckpt \
    --data-root data/womd_cache/validation $R \
    --hmm logs/responsibility/run_sdc/hmm.pkl \
    --reward-nets logs/responsibility/run_sdc/reward_nets.pt \
    --pooled-reward-net logs/responsibility/run_sdc/reward_pooled.pt \
    --first-scenario 20000 --max-scenarios 1000 \
    --out-csv logs/responsibility/run_sdc/eval.csv
```

At prediction steps 4.0/5.0/6.0/7.0 s it draws one set of K candidates per
queried agent and lets every selector pick from the same set (paired
comparison), printing mean ADE/FDE with standard errors. Agents slower than
`--min-speed` (default 2 m/s) at the prediction step are skipped, since a
standing car makes every selector look alike; the speed is kept in the CSV.

| selector | picks by |
|---|---|
| `responsibility` | reward of the level filtered causally from closed windows (the paper's method) |
| `oracle_level` | reward of the level smoothed over the whole clip (uses the future; upper bound) |
| `random_level` | reward of a random level (do the levels carry information?) |
| `pooled_reward` | one reward for all levels (does level conditioning help?) |
| `most_likely` | the model's own log-probability |
| `random`, `min_ade` | chance and the hindsight ceiling |

Rows are appended as they are computed; re-running skips finished scenarios.

## 7. Troubleshooting

| symptom | fix |
|---|---|
| `holds results computed with a different configuration` | the output directory was started with other settings; the message lists them. Use a new `--out-dir` (or `--allow-config-mismatch` if the change is known to be irrelevant) |
| CUDA out of memory in stage 1 | lower `--rollout-batch-size` or run fewer shards per GPU |
| `Can't instantiate abstract class WaymoTargetBuilderVal` | `pip install torch_geometric==2.6.1` |
| `torch_scatter`/`torch_cluster` import errors | reinstall them from the wheel index matching torch 2.4 and your `CUDA_TAG` (see `install/setup_server.sh`) |
| TensorFlow log noise | `export TF_CPP_MIN_LOG_LEVEL=3` |
| a Hydra error without a traceback | `export HYDRA_FULL_ERROR=1` |

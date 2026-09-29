#!/usr/bin/env bash
# End-to-end demo of the counterfactual-responsibility pipeline on a small
# slice of WOMD validation: every stage, the evaluation table, the ex-post
# plots and a motion-set video. Sized to finish in well under an hour on one
# GPU; scale it up with the variables below.
#
# usage (from the repository root, environment activated):
#   CKPT=ckpts/clsft_E9.ckpt DATA=data/womd_cache/validation bash scripts/responsibility/demo.sh
#
# variables (defaults in brackets):
#   CKPT        SMART checkpoint                          [ckpts/clsft_E9.ckpt]
#   DATA        cached scenario directory                 [data/womd_cache/validation]
#   OUT         output directory                          [logs/responsibility/demo]
#   N_SCEN      scenarios for stages 1-3                  [40]
#   N_EVAL      scenarios for the evaluation              [20]
#   SHARDS      stage-1 processes (spread over GPUs)      [2]
#   VIDEO_SCENARIO  scenario index for the video          [0]
#   EXTRA       extra flags for every stage, e.g. "--neighbor-future hidden"
#
# Every stage is resumable: re-running skips finished stage-1 scenarios and
# existing stage outputs (delete a file to redo that stage).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CKPT=${CKPT:-ckpts/clsft_E9.ckpt}
DATA=${DATA:-data/womd_cache/validation}
OUT=${OUT:-logs/responsibility/demo}
N_SCEN=${N_SCEN:-40}
N_EVAL=${N_EVAL:-20}
SHARDS=${SHARDS:-2}
VIDEO_SCENARIO=${VIDEO_SCENARIO:-0}
read -r -a EXTRA_ARGS <<< "${EXTRA:-}"
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore

# The SDC and the agents it interacts with, both directions; exact courtesy.
# These flags must be identical for stages 1, 3 and the evaluation.
RESP=(--query sdc --direction both --courtesy-estimator exact --neighbor-selection geometric "${EXTRA_ARGS[@]}")
MODEL=(--ckpt "$CKPT" --data-root "$DATA")
step() { printf '\n=== %s\n' "$*"; }
mkdir -p "$OUT"

step "stage 1: responsibility observations ($N_SCEN scenarios, $SHARDS shards)"
per_shard=$(( (N_SCEN + SHARDS - 1) / SHARDS ))
bash scripts/responsibility/run_sharded.sh "$SHARDS" "$OUT/stage1" -- \
  "${MODEL[@]}" "${RESP[@]}" --max-scenarios "$per_shard"

step "stage 2: HMM responsibility levels"
if [ ! -f "$OUT/hmm.pkl" ]; then
  python -m scripts.responsibility.fit_hmm --observations "$OUT/stage1" \
    --out-hmm "$OUT/hmm.pkl" --out-labeled "$OUT/labeled.pkl" --out-summary "$OUT/hmm_summary.json" \
    --n-states-grid 2 3 4 5
fi

step "ex-post plots (safety vs courtesy by level, BIC)"
python -m scripts.responsibility.ex_post_analysis --labeled-observations "$OUT/labeled.pkl" \
  --bic-summary "$OUT/hmm_summary.json" --out-dir "$OUT/plots"

step "stage 3: reward networks per level, and pooled (ablation)"
[ -f "$OUT/reward_nets.pt" ] || python -m scripts.responsibility.train_reward "${MODEL[@]}" "${RESP[@]}" \
  --labeled-observations "$OUT/labeled.pkl" --out "$OUT/reward_nets.pt"
[ -f "$OUT/reward_pooled.pt" ] || python -m scripts.responsibility.train_reward "${MODEL[@]}" "${RESP[@]}" \
  --labeled-observations "$OUT/labeled.pkl" --out "$OUT/reward_pooled.pt" --pool-levels

HELD_OUT=$(( SHARDS * per_shard ))  # stage 1 covered dataset indices [0, HELD_OUT)

step "evaluation: paired candidate selection ($N_EVAL held-out scenarios from index $HELD_OUT)"
python -m scripts.responsibility.evaluate_selection "${MODEL[@]}" "${RESP[@]}" \
  --hmm "$OUT/hmm.pkl" --reward-nets "$OUT/reward_nets.pt" --pooled-reward-net "$OUT/reward_pooled.pt" \
  --out-csv "$OUT/eval.csv" --first-scenario "$HELD_OUT" --max-scenarios "$N_EVAL" \
  | tee "$OUT/eval_summary.txt"

step "online prediction for one held-out scenario"
python -m scripts.responsibility.predict_demo "${MODEL[@]}" "${RESP[@]}" \
  --hmm "$OUT/hmm.pkl" --reward-nets "$OUT/reward_nets.pt" \
  --scenario-index "$HELD_OUT" --predict-step 50 | tee "$OUT/predict_demo.txt"

step "motion-set video"
python -m scripts.responsibility.visualize_motion_sets "${MODEL[@]}" \
  --scenario-index "$VIDEO_SCENARIO" --out-dir "$OUT/video"

step "done"
echo "results in $OUT:"
echo "  stage1/obs/*.pkl        responsibility observations per scenario"
echo "  hmm.pkl, labeled.pkl    responsibility levels"
echo "  plots/                  ex-post analysis figures"
echo "  eval.csv, eval_summary.txt  selection results"
echo "  video/motion_sets.{gif,mp4}, video/frames/"

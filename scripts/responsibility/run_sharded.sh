#!/usr/bin/env bash
# Launches stage 1 as N shard processes that fill one results directory,
# spread round-robin over the visible GPUs, each logging to OUT_DIR/logs/.
#
# usage: scripts/responsibility/run_sharded.sh NUM_SHARDS OUT_DIR -- [compute_responsibility args...]
#
#   scripts/responsibility/run_sharded.sh 4 logs/responsibility/run_sdc -- \
#       --ckpt ckpts/clsft_E9.ckpt --data-root data/womd_cache/validation \
#       --query sdc --direction both --courtesy-estimator exact
#
# Several shards per GPU are fine (and usually faster: tokenization and the
# per-window bookkeeping are CPU-bound and overlap with other shards' GPU work).
# Re-running the same command resumes: finished scenarios are skipped.
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,15p' "$0"
  exit 1
fi
NUM_SHARDS=$1
OUT_DIR=$2
shift 2
if [ "${1:-}" = "--" ]; then shift; fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi > /dev/null; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
  GPUS=("")
fi

mkdir -p "$OUT_DIR/logs"
pids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  log="$OUT_DIR/logs/shard_${i}.log"
  echo "shard $i -> GPU '${gpu:-cpu}', log $log"
  CUDA_VISIBLE_DEVICES=$gpu python -m scripts.responsibility.compute_responsibility \
    --out-dir "$OUT_DIR" --num-shards "$NUM_SHARDS" --shard-index "$i" "$@" > "$log" 2>&1 &
  pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "shard $i failed; see $OUT_DIR/logs/shard_${i}.log" >&2
    status=1
  fi
done
exit $status

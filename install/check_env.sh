#!/usr/bin/env bash
# Checks an installed environment, from the repository root:
#   bash install/check_env.sh                      # imports, GPU, unit tests
#   CKPT=ckpts/clsft_E9.ckpt DATA=data/womd_cache/validation bash install/check_env.sh
#                                                  # + counterfactual checks on real data
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
status=0

echo "=== imports and GPU"
python - <<'PY' || status=1
import torch, torch_geometric, torch_scatter, torch_cluster, lightning, hydra
print(f"torch {torch.__version__}  cuda build {torch.version.cuda}  pyg {torch_geometric.__version__}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name}  {p.total_memory / 2**30:.0f} GiB  sm_{p.major}{p.minor}")
    x = torch.randn(1024, 1024, device="cuda")
    assert torch.isfinite(x @ x).all()
    from torch_cluster import radius_graph
    radius_graph(torch.randn(100, 2, device="cuda"), r=1.0)  # compiled CUDA kernels work
else:
    print("  no CUDA device visible")
try:
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2  # noqa: F401
    print(f"tensorflow {tf.__version__}, waymo_open_dataset ok")
except Exception as e:  # only needed for preprocessing raw WOMD
    print(f"WARNING: WOMD reader unavailable ({e}); preprocessing will not work")
PY

echo "=== unit tests"
python -m pytest tests/responsibility -q -p no:cacheprovider || status=1

if [ -n "${CKPT:-}" ] && [ -n "${DATA:-}" ]; then
  echo "=== counterfactual checks against $CKPT on $DATA"
  python -m scripts.responsibility.verify_counterfactuals --ckpt "$CKPT" --data-root "$DATA" || status=1
fi

[ $status -eq 0 ] && echo "environment OK" || echo "some checks FAILED"
exit $status

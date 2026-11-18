#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
GPU="${GPU:-4}"
SOURCE_RUN="${SOURCE_RUN:-/raid/ovasqq/neugn-runs/wikics_deep_ac3_20261118_140911}"
TAG="${TAG:-wikics_pair_scratch1000_$(date +%Y%m%d_%H%M%S)}"
OUT="/raid/ovasqq/neugn-runs/$TAG"

if [[ -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT="$("$PYTHON" - "$SOURCE_RUN" <<'PY'
import json
import sys
from pathlib import Path

from models.pruning_controller import checkpoint_digest

source = Path(sys.argv[1])
expected = json.loads((source / "protocol.json").read_text())["checkpoint_digest"]
for root in (Path("checkpoints/wikics"), source):
    for path in sorted(root.rglob("model.pt")):
        if checkpoint_digest(path) == expected:
            print(path.resolve())
            raise SystemExit(0)
raise SystemExit("WikiCS checkpoint not found. Set CHECKPOINT to its path.")
PY
)"
fi

test -s "$CHECKPOINT"
mkdir -p "$OUT/queries"
cp "$SOURCE_RUN"/queries/size*.json "$OUT/queries/"
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

(
  printf 'Тег прогона: %s\nCheckpoint: %s\n' "$TAG" "$CHECKPOINT"
  git rev-parse HEAD
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
  "$PYTHON" -u -m experiments.pair_search \
    --graph wikics --checkpoint "$CHECKPOINT" --device cuda:0 \
    --output "$OUT" --sizes 10 16 24 32 \
    --from-scratch --online-states --epochs 1000 --lr 0.0003 --repeats 3 "$@"
) 2>&1 | tee "$OUT/run.log"

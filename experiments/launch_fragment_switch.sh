#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GPU:?Set GPU to the physical index of the available compute GPU}"
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
RUN_ROOT="${RUN_ROOT:-/raid/ovasqq/neugn-runs}"
SOURCE_RUN="${SOURCE_RUN:-$RUN_ROOT/wikics_deep_ac3_20261118_140911}"
CHECKPOINT="${CHECKPOINT:-checkpoints/wikics/paper1000_seed0/model.pt}"
OUT="$(mktemp -d "$RUN_ROOT/wikics_fragment_switch_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
printf 'Results: %s\n' "$OUT"
(
  git rev-parse HEAD
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  uuid="$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)"
  export CUDA_VISIBLE_DEVICES="$uuid"
  "$PYTHON" - "$CHECKPOINT" "$SOURCE_RUN/protocol.json" <<'PY'
import json
import sys
from pathlib import Path
import torch
from models.pruning_controller import checkpoint_digest
if checkpoint_digest(Path(sys.argv[1])) != json.loads(Path(sys.argv[2]).read_text())["checkpoint_digest"]:
    raise SystemExit("Checkpoint differs from source protocol")
print(torch.cuda.get_device_name(0), torch.ones(1, device="cuda:0").item(), flush=True)
PY
  "$PYTHON" -u -m experiments.fragment_switch --graph wikics --device cuda:0 \
    --query-dir "$SOURCE_RUN/queries" --checkpoint "$CHECKPOINT" --output "$OUT"
) 2>&1 | tee "$OUT/run.log"
tar -czf "$OUT.tar.gz.partial" -C "$OUT" .
mv "$OUT.tar.gz.partial" "$OUT.tar.gz"
printf 'Archive: %s.tar.gz\n' "$OUT"

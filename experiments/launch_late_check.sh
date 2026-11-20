#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GPU:?Set GPU to the physical index of a free GPU}"
py="${PYTHON:-.venv-gpu/bin/python}"
root="${RUN_ROOT:-/raid/ovasqq/neugn-runs}"
source_run="${SOURCE_RUN:-$root/wikics_deep_ac3_20261118_140911}"
checkpoint="${CHECKPOINT:-checkpoints/wikics/paper1000_seed0/model.pt}"
out=$(mktemp -d "$root/wikics_late_check_XXXXXXXX")
printf 'Results %s\n' "$out"
(
  git rev-parse HEAD
  uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
  export CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  "$py" - "$checkpoint" "$source_run/protocol.json" <<'PY'
import json
import sys
from pathlib import Path
import torch
from models.pruning_controller import checkpoint_digest
checkpoint = checkpoint_digest(Path(sys.argv[1]))
if checkpoint != json.loads(Path(sys.argv[2]).read_text())["checkpoint_digest"]:
    raise SystemExit("Checkpoint differs from source protocol")
print(torch.cuda.get_device_name(0), torch.ones(1, device="cuda:0").item(), flush=True)
PY
  "$py" -u -m experiments.late_check_pilot --graph wikics --device cuda:0 \
    --checkpoint "$checkpoint" --query-dir "$source_run/queries" --out "$out/results" "$@"
) 2>&1 | tee "$out/run.log"
tar -czf "$out.tar.gz.partial" -C "$out" results run.log
mv "$out.tar.gz.partial" "$out.tar.gz"
printf 'Archive %s.tar.gz\n' "$out"

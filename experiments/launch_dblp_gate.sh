#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GPU:?Set GPU to the physical index of the available GPU}"
py="${PYTHON:-.venv-gpu/bin/python}"
root="${RUN_ROOT:-/raid/ovasqq/neugn-runs}"
mkdir -p "$root"
out=$(mktemp -d "$root/dblp_neugn_gate_XXXXXXXX")
tag=$(basename "$out")
printf 'Results %s\n' "$out"
(
  uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
  export CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  "$py" -c 'import torch; print(torch.cuda.get_device_name(0), torch.ones(1,device="cuda:0").item())'
  git rev-parse HEAD
  if [[ -z "${CHECKPOINT:-}" ]]; then
    "$py" -u -m train.train_neugn --graph dblp_proxy15 --run "$tag" --device cuda:0 \
      --epochs "${RANK_EPOCHS:-1}" --layers 4 --batch 64 --workers 0 --save-every 1
    CHECKPOINT="checkpoints/dblp_proxy15/$tag/model.pt"
  fi
  printf 'Ranker %s\n' "$CHECKPOINT"
  "$py" -u -m experiments.dblp_gate --checkpoint "$CHECKPOINT" --out "$out/results" "$@"
) 2>&1 | tee "$out/run.log"
tar -czf "$out.tar.gz.partial" -C "$out" results run.log
mv "$out.tar.gz.partial" "$out.tar.gz"
printf 'Archive %s.tar.gz\n' "$out"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GPU:?Set GPU to a free physical GPU index}"
root="${RUN_ROOT:-/raid/ovasqq/neugn-runs}"
source_run="${SOURCE_RUN:-$root/dblp_neugn_gate_c8L8wBgG}"
py="${PYTHON:-.venv-gpu/bin/python}"
out=$(mktemp -d "$root/dblp_cut_cost_XXXXXXXX")
printf 'Results %s\n' "$out"
uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
(
  git rev-parse HEAD
  "$py" -u -m experiments.cut_cost --source "$source_run/results" --out "$out/results" "$@"
) 2>&1 | tee "$out/run.log"
tar -czf "$out.tar.gz.partial" -C "$out" results run.log
mv "$out.tar.gz.partial" "$out.tar.gz"
printf 'Archive %s.tar.gz\n' "$out"

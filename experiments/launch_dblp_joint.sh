#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GPU:?Set GPU to the physical index of an available GPU}"
py="${PYTHON:-.venv-gpu/bin/python}"
root="${RUN_ROOT:-/raid/ovasqq/neugn-runs}"
graph=dblp_proxy15
rank_epochs="${RANK_EPOCHS:-30}"
label_ac3="${LABEL_AC3:-cuda}"
train_queries="${TRAIN_QUERIES:-40}"
validation_queries="${VALIDATION_QUERIES:-16}"
test_queries="${TEST_QUERIES:-20}"
test_offset=$((train_queries + validation_queries))
query_count=$((test_offset + test_queries))

mkdir -p "$root"
out=$(mktemp -d "$root/dblp_joint_XXXXXXXX")
tag=$(basename "$out")
printf 'Results %s\n' "$out"

(
  uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
  export CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  git rev-parse HEAD
  "$py" -c 'import torch; print(torch.cuda.get_device_name(0), torch.ones(1, device="cuda:0").item())'

  "$py" -u -m queries.generate --graph "$graph" --sizes 16 24 32 \
    --seed 19 --count "$query_count" --output-dir "$out/queries"

  if [[ "${AC3_BENCH:-0}" == 1 ]]; then
    "$py" -u -m experiments.ac3_batch_bench --graph "$graph" --device cuda \
      --query-dir "$out/queries" --offset 0 --queries 4 --states 32 \
      --batches 1 4 8 --repeats 2 --output "$out/ac3_timings.csv"
  fi

  checkpoint="${CHECKPOINT:-}"
  if [[ -z "$checkpoint" ]]; then
    "$py" -u -m train.train_neugn --graph "$graph" --run "$tag" --device cuda:0 \
      --epochs "$rank_epochs" --layers 4 --batch 64 --workers 0 --save-every 1
    rank_dir="checkpoints/$graph/$tag"
    checkpoint="$rank_dir/model.pt"
    cp "$rank_dir/config.json" "$out/ranker_config.json"
    cp "$rank_dir/log.jsonl" "$out/ranker_training.jsonl"
  fi
  printf 'Ranker checkpoint %s\n' "$checkpoint"
  "$py" -u -m experiments.joint_search --graph "$graph" --device cuda:0 \
    --checkpoint "$checkpoint" --query-dir "$out/queries" --out "$out/results" \
    --seven-mode-pilot --sizes 16 24 32 --depth 10 \
    --train-queries "$train_queries" --validation-queries "$validation_queries" \
    --test-offset "$test_offset" --queries "$test_queries" \
    --label-ac3 "$label_ac3" --label-ac3-batch 8 --label-ac3-cpu-stride 8 \
    --budget 30000 --seconds 1 --repeats 3 --epochs 6 --branch-queries 0 --progress "$@"
) 2>&1 | tee "$out/run.log"

tar -czf "$out.tar.gz.partial" -C "$(dirname "$out")" "$tag"
mv "$out.tar.gz.partial" "$out.tar.gz"
printf 'Archive %s.tar.gz\n' "$out"

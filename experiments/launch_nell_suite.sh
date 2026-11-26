#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

py=${PYTHON:-.venv-gpu/bin/python}
device=${DEVICE:-cuda:0}
run=${RUN:-nell_300_seed19}
root=${RUN_ROOT:-experiments/results}
out="$root/$run"
checkpoint="checkpoints/nell_planetoid/$run/model.pt"
epochs=${EPOCHS:-300}
train_queries=${TRAIN_QUERIES:-60}
validation_queries=${VALIDATION_QUERIES:-20}
test_queries=${TEST_QUERIES:-30}
test_offset=${TEST_OFFSET:-80}
query_count=$((test_offset + test_queries))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

if [[ "$device" == cuda* && -n "${GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader)
fi
mkdir -p "$out"
phase=${PHASE:-all}
case "$phase" in
    all|train|eval) ;;
    *) printf 'PHASE must be all, train or eval\n' >&2; exit 2 ;;
esac

if [[ "$phase" != train ]]; then
    "$py" -u -m queries.generate --graph nell_planetoid --sizes 16 24 32 --count "$query_count" \
        --seed 19 --output-dir "$out/queries" 2>&1 | tee "$out/queries.log"
fi
if [[ "$phase" != eval ]]; then
    extra=()
    if [[ -n "${STEPS_PER_EPOCH:-}" ]]; then
        extra+=(--steps-per-epoch "$STEPS_PER_EPOCH")
    fi
    "$py" -u -m train.train_neugn --graph nell_planetoid --run "$run" --resume \
        --device "$device" --epochs "$epochs" --batch "${BATCH:-128}" --layers 4 --seed 19 \
        --workers "${WORKERS:-4}" --save-every "${SAVE_EVERY:-1}" \
        ${extra[@]+"${extra[@]}"} 2>&1 | tee -a "$out/train.log"
fi
if [[ "$phase" != train ]]; then
    eval_dir=$(mktemp -d "$out/eval.XXXXXX")
    "$py" -u -m experiments.triple_search --nell-suite --graph nell_planetoid --device "$device" \
        --checkpoint "$checkpoint" --query-dir "$out/queries" --out "$eval_dir/results" \
        --sizes 16 24 32 --train-queries "$train_queries" --validation-queries "$validation_queries" \
        --test-offset "$test_offset" --queries "$test_queries" --budget "${DFS_STEPS:-30000}" --seconds 1 \
        --epochs "${SELECTOR_EPOCHS:-6}" --reward-epochs "${REWARD_EPOCHS:-3}" \
        --joint-epochs "${JOINT_EPOCHS:-6}" --repeats "${REPEATS:-2}" 2>&1 | tee "$eval_dir/run.log"
    printf 'Results: %s/results\n' "$eval_dir"
fi

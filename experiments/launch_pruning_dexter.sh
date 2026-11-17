#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

budget=${PRUNING_BUDGET:-20000}
networkx_version=${PRUNING_NETWORKX_VERSION:-3.7}

if [[ ${1:-} == --run && $# -le 2 ]]; then
    limit=0
    case ${2:-} in
        '') ;;
        --smoke) limit=3 ;;
        *) echo 'Usage: launch_pruning_dexter.sh --run [--smoke]' >&2; exit 2 ;;
    esac
    if [[ ! $budget =~ ^[1-9][0-9]*$ ]]; then
        echo 'PRUNING_BUDGET must be a positive integer.' >&2
        exit 2
    fi
    printf 'running\n' > run.status
    trap 'status=$?; printf "exit_status=%s\n" "$status" > run.status' EXIT

    usable_python() {
        "$1" -c 'import sys, networkx, numpy, torch; assert sys.version_info >= (3, 11); assert networkx.__version__ == sys.argv[1]' \
            "$networkx_version" >/dev/null 2>&1
    }

    python=""
    for candidate in "$PWD/.venv/bin/python" "$HOME/subgraph_isomorphism/.venv/bin/python" python3; do
        if usable_python "$candidate"; then
            python=$candidate
            break
        fi
    done
    if [[ -z $python ]]; then
        for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
            if "$candidate" -c 'import sys; assert sys.version_info >= (3, 11)' >/dev/null 2>&1; then
                "$candidate" -m venv .venv
                python="$PWD/.venv/bin/python"
                break
            fi
        done
        if [[ -z $python ]]; then
            echo 'Python 3.11 or newer is required on Dexter.' >&2
            exit 1
        fi
        "$python" -m pip install numpy "networkx==$networkx_version"
        "$python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi

    export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    "$python" -c 'import platform, sys, networkx, numpy, torch; print(platform.node(), sys.version, "networkx", networkx.__version__, "numpy", numpy.__version__, "torch", torch.__version__)'

    modes=(none arc domains motifs)
    for graph in hamster rewired; do
        query_sets=(data/queries/"$graph"/size*.json)
        limit_args=()
        if [[ $limit != 0 ]]; then
            query_sets=(data/queries/"$graph"/size16_dense.json)
            limit_args=(--limit "$limit")
        fi
        for query_set in "${query_sets[@]}"; do
            name=${query_set##*/}
            size=${name#size}
            size=${size%%_*}
            density=${name#*_}
            density=${density%.json}
            for mode in "${modes[@]}"; do
                printf '%s %s size=%s density=%s pruning=%s\n' \
                    "$(date -u +%FT%TZ)" "$graph" "$size" "$density" "$mode"
                "$python" -m experiments.fms --graph "$graph" --size "$size" --density "$density" \
                    --query-set "$query_set" --method neugn --device cpu --pruning "$mode" \
                    --budget "$budget" --checkpoint checkpoints/hamster/paper/model.pt \
                    ${limit_args[@]+"${limit_args[@]}"}
            done
        done
    done
    exit
fi

if [[ ${1:-} != --bundle || $# != 2 || $2 != /* ]]; then
    printf 'Usage: %s --bundle /absolute/path.tar.gz | --run [--smoke]\n' "$0" >&2
    exit 2
fi
bundle=$2
if [[ -e $bundle ]]; then
    printf 'Archive already exists: %s\n' "$bundle" >&2
    exit 1
fi

files=(pyproject.toml experiments/__init__.py experiments/fms.py experiments/launch_pruning_dexter.sh
    models/*.py queries/*.py search/*.py train/*.py data/*.py
    data/queries/hamster/size*.json data/queries/rewired/size*.json
    data/raw/petster-hamster.tar.bz2 checkpoints/hamster/paper/model.pt)
for file in "${files[@]}"; do
    if [[ ! -r $file ]]; then
        printf 'Missing %s\n' "$file" >&2
        exit 1
    fi
done

COPYFILE_DISABLE=1 tar -czf "$bundle" "${files[@]}"
printf '%s\n' "$bundle"

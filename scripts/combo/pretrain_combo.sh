#!/bin/bash

set -e

MAX_JOBS=4

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -job) MAX_JOBS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

EPOCHS=10000
LR=1e-4
HIDDEN=256
pids=()

run_job() {
    "$@" &
    pids+=($!)
    if [[ ${#pids[@]} -ge $MAX_JOBS ]]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
}

run_all_variants() {
    local name=$1
    local path="modril/expert_datasets/${name}.pt"
    echo ">>> [FULL] $name"

    run_job python deps/baselines/ebil/deen.py --traj-load-path "$path" --num-epoch "$EPOCHS" --lr "$LR" --hidden-dim "$HIDDEN"
    run_job python modril/flowril/pretrain.py --traj-load-path "$path" --num-epoch "$EPOCHS" --lr "$LR" --hidden-dim "$HIDDEN"
    run_job python modril/flowril/pretrain.py --traj-load-path "$path" --num-epoch "$EPOCHS" --lr "$LR" --hidden-dim "$HIDDEN" --enable-loss-stable false --enable-loss-anti false
    run_job python modril/flowril/pretrain.py --traj-load-path "$path" --num-epoch "$EPOCHS" --lr "$LR" --hidden-dim "$HIDDEN" --enable-loss-anti false
    run_job python modril/flowril/pretrain.py --traj-load-path "$path" --num-epoch "$EPOCHS" --lr "$LR" --hidden-dim "$HIDDEN" --enable-loss-stable false
    run_job python modril/flowril/pretrain.py --traj-load-path "$path" --num-epoch "$EPOCHS" --lr "$LR" --hidden-dim "$HIDDEN" --option 2fs
}

run_basic_variants() {
    local name=$1
    local path="modril/expert_datasets/${name}.pt"
    echo ">>> [BASIC] $name"

    run_job python deps/baselines/ebil/deen.py --traj-load-path "$path" --num-epoch "$EPOCHS" --lr "$LR" --hidden-dim "$HIDDEN"
    run_job python modril/flowril/pretrain.py --traj-load-path "$path" --num-epoch "$EPOCHS" --lr "$LR" --hidden-dim "$HIDDEN"
    run_job python modril/flowril/pretrain.py --traj-load-path "$path" --num-epoch "$EPOCHS" --lr "$LR" --hidden-dim "$HIDDEN" --option 2fs
}

# === Run combo tasks ===
run_all_variants sine
#run_all_variants maze
#run_all_variants pick
#run_all_variants walker
#run_all_variants push
#run_all_variants halfcheetach
#run_all_variants hand
#run_all_variants ant

for pid in "${pids[@]}"; do
    wait "$pid"
done
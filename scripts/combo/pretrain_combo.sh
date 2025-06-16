#!/bin/bash

set -e
MAX_JOBS=1

EPOCHS=10000
LR=1e-4
HIDDEN=256
job_count=0

run_job() {
    "$@" &
    ((job_count+=1))
    if [[ $job_count -ge $MAX_JOBS ]]; then
        wait
        job_count=0
    fi
}

run_all_variants() {
    local name=$1
    local path="modril/expert_datasets/${name}.pt"

    echo ">>> [FULL] $name"

    run_job python deps/baselines/ebil/deen.py --traj-load-path $path --num-epoch $EPOCHS --lr $LR --hidden-dim $HIDDEN
    run_job python modril/flowril/pretrain.py --traj-load-path $path --num-epoch $EPOCHS --lr $LR --hidden-dim $HIDDEN
    run_job python modril/flowril/pretrain.py --traj-load-path $path --num-epoch $EPOCHS --lr $LR --hidden-dim $HIDDEN --enable-loss-anti false
    run_job python modril/flowril/pretrain.py --traj-load-path $path --num-epoch $EPOCHS --lr $LR --hidden-dim $HIDDEN --enable-loss-stable false
    run_job python modril/flowril/pretrain.py --traj-load-path $path --num-epoch $EPOCHS --lr $LR --hidden-dim $HIDDEN --enable-loss-stable false --enable-loss-anti false
    run_job python modril/flowril/pretrain.py --traj-load-path $path --num-epoch $EPOCHS --lr $LR --hidden-dim $HIDDEN --option 2fs
}

run_basic_variants() {
    local name=$1
    local path="modril/expert_datasets/${name}.pt"

    echo ">>> [BASIC] $name"

    run_job python deps/baselines/ebil/deen.py --traj-load-path $path --num-epoch $EPOCHS --lr $LR --hidden-dim $HIDDEN
    run_job python modril/flowril/pretrain.py --traj-load-path $path --num-epoch $EPOCHS --lr $LR --hidden-dim $HIDDEN
    run_job python modril/flowril/pretrain.py --traj-load-path $path --num-epoch $EPOCHS --lr $LR --hidden-dim $HIDDEN --option 2fs
}

# === Run combo tasks ===
run_all_variants sine
#run_all_variants maze
#run_all_variants pick
#run_all_variants walker
#run_basic_variants push
#run_basic_variants halfcheetach
#run_basic_variants hand
#run_basic_variants ant

wait
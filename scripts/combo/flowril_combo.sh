#!/bin/bash

task_filter=""
method_filter=""
job_limit=4

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -task) task_filter="$2"; shift ;;
        -method) method_filter="$2"; shift ;;
        -job) job_limit="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

declare -A task_methods
task_methods[sine]="flowril flowril-2fs"
task_methods[maze]="flowril flowril-2fs flowril-no-a flowril-no-s flowril-no-as"
task_methods[pick]="flowril flowril-2fs flowril-no-a flowril-no-s flowril-no-as"
task_methods[walker]="flowril flowril-2fs flowril-no-a flowril-no-s flowril-no-as"
task_methods[push]="flowril flowril-2fs"
task_methods[halfcheetach]="flowril flowril-2fs"
task_methods[hand]="flowril flowril-2fs"
task_methods[ant]="flowril flowril-2fs"

all_tasks=(sine maze pick walker push halfcheetach hand ant)

configs=()
for task in "${all_tasks[@]}"; do
    if [[ -n "$task_filter" && "$task" != "$task_filter" ]]; then
        continue
    fi

    methods=(${task_methods[$task]})
    for method in "${methods[@]}"; do
        if [[ -n "$method_filter" && "$method" != "$method_filter" ]]; then
            continue
        fi
        config="./configs/${task}/${method}.yaml"
        if [[ -f "$config" ]]; then
            configs+=("$config")
        fi
    done
done

running_jobs=0
for config in "${configs[@]}"; do
    echo "Running $config"
    ./scripts/run.sh "$config" &
    ((running_jobs++))

    if (( running_jobs >= job_limit )); then
        wait -n
        ((running_jobs--))
    fi
done

wait
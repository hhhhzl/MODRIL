#!/bin/bash

pip install -U gdown
current_directory=$(dirname "$(realpath "$0")")
expert_datasets_path="$current_directory/../modril/expert_datasets"
python "${current_directory}/../modril/utils/download_demos.py" --dir "$expert_datasets_path"
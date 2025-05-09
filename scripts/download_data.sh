#!/bin/bash

pip install -U gdown
current_directory=$(dirname "$(realpath "$0")")
expert_datasets_path="$current_directory/../mdril/expert_datasets"
python "${current_directory}/../mdril/utils/download_demos.py" --dir "$expert_datasets_path"
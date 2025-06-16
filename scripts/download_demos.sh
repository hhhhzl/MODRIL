if [[ "$1" != "exp" && "$1" != "full" ]]; then
  echo "Usage: ./download_demos.sh [exp|full]"
  exit 1
fi

version="$1"

pip install -U gdown

current_directory=$(dirname "$(realpath "$0")")
expert_datasets_path="$current_directory/../modril/expert_datasets"

python ${current_directory}/../modril/utils/download_demos.py --dir "$expert_datasets_path" --v "$version"


if [[ "$version" == "full" ]]; then
  python ${current_directory}/../modril/utils/clip_pick.py
  python ${current_directory}/../modril/utils/clip_push.py
fi
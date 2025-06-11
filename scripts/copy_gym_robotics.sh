#!/bin/bash

current_directory=$(dirname "$(realpath "$0")")
libpath="${current_directory}/../deps/gym17/robotics"

current_directory="$(dirname "$(realpath "$0")")"
libpath="${current_directory}/../deps/gym17/robotics"
gym_root="$(python3 - <<'PYCODE'
import gym, os
print(os.path.dirname(gym.__file__))
PYCODE
)"
target_dir="${gym_root}/envs/robotics"

echo "Detected gym root: $gym_root"
echo "Copying from    : $libpath"
echo "Copying to      : $target_dir"
mkdir -p "$target_dir"
cp -rf "${libpath}/." "$target_dir/"

echo "✔ robotics copied to $target_dir"
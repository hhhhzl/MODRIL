#!/bin/bash

MUJOCO_DIR="$HOME/.mujoco"
TAR_PATH="../assets/mujoco/macos/mujoco210-macos-x86_64.tar.gz"
EXTRACTED_DIR="$MUJOCO_DIR/mujoco210-macos-x86_64"
TARGET_DIR="$MUJOCO_DIR/mujoco210"

# Create MuJoCo directory
mkdir -p "$MUJOCO_DIR"
tar -xf "$TAR_PATH" -C "$MUJOCO_DIR"
if [ -d "$EXTRACTED_DIR" ]; then
    mv "$EXTRACTED_DIR" "$TARGET_DIR"
fi
cp ../assets/mujoco/macos/mjkey.txt "$MUJOCO_DIR/"
cp ../assets/mujoco/macos/mjkey.txt "$TARGET_DIR/bin/"


if [ -n "$ZSH_VERSION" ] || [ "$(basename "$SHELL")" = "zsh" ]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.bashrc"
fi
touch "$SHELL_RC"
if ! grep -q 'MUJOCO_KEY_PATH' "$SHELL_RC"; then
    echo '' >> "$SHELL_RC"
    echo '# MuJoCo environment setup' >> "$SHELL_RC"
    echo 'export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH' >> "$SHELL_RC"
    echo 'export MUJOCO_KEY_PATH=$HOME/.mujoco/mjkey.txt' >> "$SHELL_RC"
fi
source "$SHELL_RC"

echo "MuJoCo Install Successfully! Run simulation To test"

# Run simulation
# shellcheck disable=SC2164
#cd ~/.mujoco/mujoco210/bin
#./simulate ../model/humanoid.xml
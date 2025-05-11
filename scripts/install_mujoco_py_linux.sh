#!/bin/bash

apt update
apt-get install patchelf
apt-get install python3-dev build-essential libssl-dev libffi-dev libxml2-dev
apt-get install libxslt1-dev zlib1g-dev libglew1.5 libglew-dev python3-pip

cd ~/.mujoco
git clone https://github.com/openai/mujoco-py
cd mujoco-py
pip install -r requirements.txt
pip install -r requirements.dev.txt
pip3 install -e . --no-cache

apt install libosmesa6-dev libgl1-mesa-glx libglfw3
ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
# If you get an error like: "ln: failed to create symbolic link '/usr/lib/x86_64-linux-gnu/libGL.so': File exists", it's okay to proceed
pip3 install -U 'mujoco-py<2.2,>=2.1'

echo "MuJoCo-py Install Successfully! Run Example To Test."

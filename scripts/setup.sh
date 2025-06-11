#!/bin/bash

cd ../deps/rl-toolkit
pip install -e .

cd ../d4rl
cd d4rl
pip install -e .

cd ../..
pip install -r requirements.txt
pip install -e .
# Flow Imitation Learning

---

## Introduction
FLOWRIL: Flow-Matching Log-Density Occupancy With Reward-Guided Imitation Learning via Stable Coupled Residuals

## Installation

#### Env Requirements: 
  - Python = 3.10
  - ubuntu = 22.04
  - cuda = 11.8.0
  - pytorch = 2.1.0
#### Install Mujoco && Mujoco-py (only support for Linux):
```
./scripts/install_mujoco_linux.sh
./scripts/install_mujoco_py_linux.sh
```
Note: make sure your "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin" is activate by
```
source ~/.bashrc
```
#### Environment Setup:
```
./scripts/setup.sh
```
#### Load gym robotics:
```
./scripts/copy_gym_robotics.sh
```
#### Check for Important Dependencies Version: 
  - gym == 0.23.1
  - cython == 0.29.22
  - mujoco-py == 2.1.2.14
  - mujoco == 3.3.2

## Datasets & Wandb

#### Download Demo Datasets
```
./scripts/download_demos.sh
```

#### Set up Wandb
```
wandb login <YOUR_WANDB_APIKEY>
```

#### Test Mojuco && Initial GLFW
```
cd ~/.mujoco/mujoco210/bin
xvfb-run -s "-screen 0 1024x768x24" ./simulate ../model/humanoid.xml
```
If you see no error, which means it successfully runs in the virtual screen.

## Reproduce Our Experiments



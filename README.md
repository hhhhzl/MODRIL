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
#### Environment Setup:
```
./scripts/setup.sh
```
#### Check for Important Dependencies Version: 
  - gym == 0.23.1
  - cython == 0.29.22
  - mujoco-py == 2.1.2.14
  - mojoco == 3.3.2

## Datasets & Wandb

#### Download Demo Datasets
```
./scripts/download_demos.sh
```

#### Set up Wandb
```
wandb login <YOUR_WANDB_APIKEY>
```

## Reproduce Our Experiments



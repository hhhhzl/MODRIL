import torch
import os

dir = os.path.dirname(os.path.realpath(__file__))

traj = torch.load(os.path.join(dir, "..", "expert_datasets/pick_100.pt"))
print(traj["obs"].shape)  # ! 20311
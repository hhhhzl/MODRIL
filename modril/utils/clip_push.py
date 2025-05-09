import torch
import os

dir = os.path.dirname(os.path.realpath(__file__))

traj = torch.load(os.path.join(dir, "..", "expert_datasets/push_partial2.pt"))
print(traj["obs"].shape)  # ! 20311


def clip_dataset(data, num_transition=10000):
    trajs = {}
    for k, v in data.items():
        print(k)
        print(v.shape)
        trajs[k] = v[:min(num_transition, len(v))]
    num_trajs = trajs['done'].sum() // 2
    print(f'Number of trajs: {num_trajs}')
    print(f'Number of transitions: {min(num_transition, len(v))}')
    return trajs


target_list = [2000, 5000, 10000]

for target in target_list:
    trajs = clip_dataset(traj, target)
    torch.save(trajs, os.path.join(dir, "..", f'expert_datasets/push_{target}.pt'))
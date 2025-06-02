import torch
import math
import numpy as np
import torch.nn as nn


def normalize(x):
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True).clip(min=1e-6)
    return (x - mu) / std, mu, std


def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def norm_state(x):
    return (x - 5.0) / 5.0 * np.pi


def denorm_state(z):
    return z * 5.0 / np.pi + 5.0


def timestep_embed(t, dim):
    """"""
    half = dim // 2
    emb = torch.exp(
        torch.arange(half, dtype=torch.float32, device=t.device)
        * -(math.log(10000.0) / (half - 1))
    )
    emb = t[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


def compute_advantage(gamma, lmbda, td_delta):
    td = td_delta.detach()
    advantages = torch.zeros_like(td)
    last_adv = torch.zeros(1, device=td.device, dtype=td.dtype)
    for t in reversed(range(td.shape[0])):
        last_adv = td[t] + gamma * lmbda * last_adv
        advantages[t] = last_adv
    return advantages


def random_sample_from_intervals(intervals):
    interval = intervals[torch.randint(len(intervals), (1,))]
    return torch.rand(1) * (interval[1] - interval[0]) + interval[0]


def sample_expert_ground_truth(num, min=0, max=10, split=100):
    intervals = np.arange(min, max, (max - min) / split).reshape(-1, 2).tolist()
    x = torch.stack(
        [random_sample_from_intervals(intervals) for _ in range(num)]
    ).squeeze()
    return x

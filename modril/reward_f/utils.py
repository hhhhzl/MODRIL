import torch
import math
import numpy as np
import torch.nn as nn


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
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

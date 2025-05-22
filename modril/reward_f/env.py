import gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from rlf.algos import PPO
import wandb
import datetime


def norm_state(x_raw):
    return (x_raw - 5.0) / 5.0 * np.pi


def denorm_state(x_norm):
    return x_norm * 5.0 / np.pi + 5.0


class Environment:
    def __init__(self, data_raw, x_raw):
        """
        data_raw : ndarray  shape (N,2) ,  [:,0] = x_raw , [:,1] = y
        x_raw    : ndarray
        """
        self.x_raw = x_raw  # [0,10]
        self.x_norm = norm_state(x_raw)  # (−π,π)
        self.data_raw = data_raw  # 同上
        self.tolerance = 1e-3

    def _raw_to_norm(self, x):
        return norm_state(x)

    def _norm_to_raw(self, z):
        return denorm_state(z)

    def get_true_y(self, x_norm):
        x_raw = self._norm_to_raw(np.asarray(x_norm))
        idx = np.abs(self.x_raw - x_raw[..., None]).argmin(axis=-1)
        return self.data_raw[idx, 1]

    def reset(self):
        idx = np.random.randint(len(self.x_norm))
        return float(self.x_norm[idx])

    def step(self, state_norm, predicted_y):
        true_y = self.get_true_y(state_norm)
        next_state = self.reset()
        return next_state, true_y

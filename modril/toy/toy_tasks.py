import numpy as np
from modril.toy.utils import normalize
from modril.toy.env import Environment


# Function/task abstraction
# --- Task Abstraction ---
class TaskBase:
    def sample_expert(self, n_samples=None):
        raise NotImplementedError

    @property
    def state_dim(self):
        raise NotImplementedError

    @property
    def action_dim(self):
        raise NotImplementedError


class Sine1D(TaskBase):
    def __init__(self, amplitude=1.0, freq=0.1, scale=20.0, phase=0.0, noise_std=0.05, n_points=1000):
        # generate expert trajectory
        x = np.linspace(0, 10, n_points)
        y = amplitude * np.sin(scale * freq * np.pi * x + phase)
        y += np.random.randn(n_points) * noise_std
        # normalize state and action
        s_norm, self.s_mu, self.s_std = normalize(x[:, None])
        a_norm, self.a_mu, self.a_std = normalize(y[:, None])
        self.expert_s = s_norm
        self.expert_a = a_norm
        data_raw = np.hstack([s_norm, a_norm])
        # define environment if needed
        self.env = Environment(data_raw, x)

    @property
    def state_dim(self):
        return 1

    @property
    def action_dim(self):
        return 1

    def sample_expert(self, n_samples=None):
        return np.hstack([self.expert_s, self.expert_a])


class TwoMoons2D(TaskBase):
    def __init__(self, n_samples=5000, r=2.0, noise=0.1):
        t = np.random.rand(n_samples) * np.pi
        x1 = r * np.cos(t) + np.random.randn(n_samples) * noise
        y1 = r * np.sin(t) + np.random.randn(n_samples) * noise
        x2 = r * np.cos(t) + r + np.random.randn(n_samples) * noise
        y2 = -r * np.sin(t) + np.random.randn(n_samples) * noise
        data = np.vstack([np.stack([x1, y1], 1), np.stack([x2, y2], 1)])
        self.expert_s = data  # for 2D, treat data as states
        self.expert_a = None
        self.env = None

    @property
    def state_dim(self):
        return 2

    @property
    def action_dim(self):
        return 0

    def sample_expert(self, n_samples=None):
        return self.expert_s

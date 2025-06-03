import numpy as np
from modril.toy.utils import normalize, sample_expert_ground_truth
from modril.toy.env import Environment1DStatic, Environment2D, Environment1DDynamic


# Function/task abstraction
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
    """
    Single-frequency sine: y = A * sin(ω x + φ) + noise
    """

    def __init__(self, amplitude=1.0, freq=0.1, scale=2.0, phase=0.0, noise_std=0.05, n_points=1000, split=20):
        # x = np.linspace(0, 10, n_points)[:, None]
        x_torch = sample_expert_ground_truth(n_points, 0, 10, split)
        x = x_torch.unsqueeze(-1).numpy()

        y = amplitude * np.sin(scale * freq * np.pi * x + phase)[:, 0]
        y = y + np.random.randn(n_points) * noise_std
        s_norm, self.s_mu, self.s_std = normalize(x)
        a_norm, self.a_mu, self.a_std = normalize(y[:, None])
        self.expert_s = s_norm
        self.expert_a = a_norm
        self.x = x
        self.env = Environment1DDynamic(np.hstack([self.expert_s, self.expert_a]), self.x, self.state_dim, self.action_dim)

    @property
    def state_dim(self):
        return 1

    @property
    def action_dim(self):
        return 1

    def sample_expert(self, n_samples=None):
        return np.hstack([self.expert_s, self.expert_a])


class MultiSine1D(TaskBase):
    """
    Multi-frequency sine: y = sin(x) + 0.5 sin(3x) + noise
    """

    def __init__(self, noise_std=0.05, n_points=1000):
        x = np.linspace(0, 10, n_points)[:, None]
        y = np.sin(x)[:, 0] + 0.5 * np.sin(3 * x)[:, 0]
        y = y + np.random.randn(n_points) * noise_std
        s_norm, self.s_mu, self.s_std = normalize(x)
        a_norm, self.a_mu, self.a_std = normalize(y[:, None])
        self.expert_s = s_norm
        self.expert_a = a_norm
        self.env = Environment1DStatic(np.hstack([self.expert_s, self.expert_a]), x, self.state_dim, self.action_dim)

    @property
    def state_dim(self):
        return 1

    @property
    def action_dim(self):
        return 1

    def sample_expert(self, n_samples=None):
        return np.hstack([self.expert_s, self.expert_a])


class GaussSine1D(TaskBase):
    """
    Gaussian-envelope sine: y = exp(-x^2) * sin(2x) + noise
    """

    def __init__(self, noise_std=0.02, n_points=1000, range_xy=3.0):
        x = np.linspace(-range_xy, range_xy, n_points)[:, None]
        y = np.exp(-x ** 2)[:, 0] * np.sin(2 * x)[:, 0]
        y = y + np.random.randn(n_points) * noise_std
        s_norm, self.s_mu, self.s_std = normalize(x)
        a_norm, self.a_mu, self.a_std = normalize(y[:, None])
        self.expert_s = s_norm
        self.expert_a = a_norm
        self.env = Environment1DStatic(np.hstack([self.expert_s, self.expert_a]), x, self.state_dim, self.action_dim)

    @property
    def state_dim(self):
        return 1

    @property
    def action_dim(self):
        return 1

    def sample_expert(self, n_samples=None):
        return np.hstack([self.expert_s, self.expert_a])


class Poly1D(TaskBase):
    """
    Polynomial: y = 0.1 x^3 - 0.5 x + noise
    """

    def __init__(self, noise_std=0.05, n_points=1000, range_xy=3.0):
        x = np.linspace(-range_xy, range_xy, n_points)[:, None]
        y = 0.1 * x[:, 0] ** 3 - 0.5 * x[:, 0]
        y = y + np.random.randn(n_points) * noise_std
        s_norm, self.s_mu, self.s_std = normalize(x)
        a_norm, self.a_mu, self.a_std = normalize(y[:, None])
        self.expert_s = s_norm
        self.expert_a = a_norm
        self.env = Environment1DStatic(np.hstack([self.expert_s, self.expert_a]), x, self.state_dim, self.action_dim)

    @property
    def state_dim(self):
        return 1

    @property
    def action_dim(self):
        return 1

    def sample_expert(self, n_samples=None):
        return np.hstack([self.expert_s, self.expert_a])


class GaussianHill2D(TaskBase):
    """
    2D surface: z = exp(- (x^2 + y^2) )
    """

    def __init__(self, nx=100, ny=100, range_xy=3.0):
        xs = np.linspace(-range_xy, range_xy, nx)
        ys = np.linspace(-range_xy, range_xy, ny)
        X, Y = np.meshgrid(xs, ys)
        Z = np.exp(-(X ** 2 + Y ** 2))
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        s = pts[:, :2]
        a = pts[:, 2:3]
        s_norm, self.s_mu, self.s_std = normalize(s)
        a_norm, self.a_mu, self.a_std = normalize(a)
        self.expert_s = s_norm
        self.expert_a = a_norm
        self.env = Environment2D(self.expert_s, self.expert_a, self.state_dim, self.action_dim)

    @property
    def state_dim(self):
        return 2

    @property
    def action_dim(self):
        return 1

    def sample_expert(self, n_samples=None):
        return np.hstack([self.expert_s, self.expert_a])


class MexicanHat2D(TaskBase):
    """
    2D surface: f(x,y) = (1 - r^2) * exp(-r^2 / 2)
    """

    def __init__(self, nx=100, ny=100, range_xy=3.0):
        xs = np.linspace(-range_xy, range_xy, nx)
        ys = np.linspace(-range_xy, range_xy, ny)
        X, Y = np.meshgrid(xs, ys)
        R2 = X ** 2 + Y ** 2
        Z = (1 - R2) * np.exp(-R2 / 2)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        s = pts[:, :2]
        a = pts[:, 2:3]
        s_norm, self.s_mu, self.s_std = normalize(s)
        a_norm, self.a_mu, self.a_std = normalize(a)
        self.expert_s = s_norm
        self.expert_a = a_norm
        self.env = Environment2D(self.expert_s, self.expert_a, self.state_dim, self.action_dim)

    @property
    def state_dim(self):
        return 2

    @property
    def action_dim(self):
        return 1

    def sample_expert(self, n_samples=None):
        return np.hstack([self.expert_s, self.expert_a])


class Saddle2D(TaskBase):
    """
    2D surface: z = x^2 - y^2
    """

    def __init__(self, nx=100, ny=100, range_xy=3.0):
        xs = np.linspace(-range_xy, range_xy, nx)
        ys = np.linspace(-range_xy, range_xy, ny)
        X, Y = np.meshgrid(xs, ys)
        Z = X ** 2 - Y ** 2
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        s = pts[:, :2]
        a = pts[:, 2:3]
        s_norm, self.s_mu, self.s_std = normalize(s)
        a_norm, self.a_mu, self.a_std = normalize(a)
        self.expert_s = s_norm
        self.expert_a = a_norm
        self.env = Environment2D(self.expert_s, self.expert_a, self.state_dim, self.action_dim)

    @property
    def state_dim(self):
        return 2

    @property
    def action_dim(self):
        return 1

    def sample_expert(self, n_samples=None):
        return np.hstack([self.expert_s, self.expert_a])


class SinusoidalRipple2D(TaskBase):
    """
    2D surface: z = sin(3x) * sin(3y)
    """

    def __init__(self, nx=100, ny=100, range_xy=3.0):
        xs = np.linspace(-range_xy, range_xy, nx)
        ys = np.linspace(-range_xy, range_xy, ny)
        X, Y = np.meshgrid(xs, ys)
        Z = np.sin(3 * X) * np.sin(3 * Y)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        s = pts[:, :2]
        a = pts[:, 2:3]
        s_norm, self.s_mu, self.s_std = normalize(s)
        a_norm, self.a_mu, self.a_std = normalize(a)
        self.expert_s = s_norm
        self.expert_a = a_norm
        self.env = Environment2D(self.expert_s, self.expert_a, self.state_dim, self.action_dim)

    @property
    def state_dim(self):
        return 2

    @property
    def action_dim(self):
        return 1

    def sample_expert(self, n_samples=None):
        return np.hstack([self.expert_s, self.expert_a])


class BimodalGaussian2D(TaskBase):
    """
    2D surface: z = N((-2,-2),I) + N((2,2),I)
    """

    def __init__(self, nx=100, ny=100, range_xy=3.0):
        xs = np.linspace(-range_xy, range_xy, nx)
        ys = np.linspace(-range_xy, range_xy, ny)
        X, Y = np.meshgrid(xs, ys)
        Z1 = np.exp(-((X + 2) ** 2 + (Y + 2) ** 2))
        Z2 = np.exp(-((X - 2) ** 2 + (Y - 2) ** 2))
        Z = Z1 + Z2
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        s = pts[:, :2]
        a = pts[:, 2:3]
        s_norm, self.s_mu, self.s_std = normalize(s)
        a_norm, self.a_mu, self.a_std = normalize(a)
        self.expert_s = s_norm
        self.expert_a = a_norm
        self.env = Environment2D(self.expert_s, self.expert_a, self.state_dim, self.action_dim)

    @property
    def state_dim(self):
        return 2

    @property
    def action_dim(self):
        return 1

    def sample_expert(self, n_samples=None):
        return np.hstack([self.expert_s, self.expert_a])

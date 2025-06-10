import numpy as np
from modril.toy.utils import norm_state, denorm_state
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym import error, logger, spaces
from gym.utils import seeding
import torch
from gym import Env
from gym.spaces import Box
from collections import deque

# def mujoco_seed(self, seed=None):
#     self.np_random, seed = seeding.np_random(seed)
#     return [seed]

# setattr(MujocoEnv, 'seed', mujoco_seed)


class Environment1DDynamic:
    def __init__(
            self,
            data_raw: np.ndarray,
            x_raw: np.ndarray,
            state_dim: int,
            action_dim: int,
            dt: float = 0.1,
            horizon: int = 100
    ):
        assert state_dim == 1 and action_dim == 1
        self.x_raw = x_raw
        self.x_norm = norm_state(x_raw)
        self.data_raw = data_raw
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.dt = dt
        self.horizon = horizon
        self.cur_step = 0
        self.state = None

    def _raw_to_norm(self, x):
        return norm_state(x)

    def _norm_to_raw(self, z):
        return denorm_state(z)

    def get_true_y(self, x_norm_scalar: float) -> float:
        x_raw = self._norm_to_raw(np.asarray(x_norm_scalar))
        idx = np.abs(self.x_raw.reshape(-1) - x_raw).argmin(axis=-1)
        return float(self.data_raw[idx, 1])

    def reset(self, batch_size: int = 1) -> np.ndarray:
        self.cur_step = 0
        states = np.random.choice(self.x_norm.reshape(-1), size=batch_size)
        self.state = states.copy().astype(np.float32).reshape(-1)
        return self.state.copy()

    def step(self, predicted_y: np.ndarray):
        a_arr = np.asarray(predicted_y, dtype=np.float32).reshape(-1)
        assert a_arr.ndim == 1 and a_arr.shape[0] == 1
        s_arr = self.state.reshape(-1)
        s_scalar = float(s_arr[0])
        a_scalar = float(a_arr[0])
        true_y = self.get_true_y(s_scalar)
        r_t = - (a_scalar - true_y) ** 2
        next_s_scalar = float(np.clip(s_scalar + a_scalar * self.dt, -1.0, 1.0))
        self.state = np.array([next_s_scalar], dtype=np.float32)
        self.cur_step += 1
        done = False
        if self.cur_step >= self.horizon:
            done = True

        info = {"true_y": true_y}
        return np.array([next_s_scalar], dtype=np.float32), float(r_t), done, info

    def batch_step(self, states_norm: np.ndarray, predicted_y: np.ndarray):
        states = np.asarray(states_norm, dtype=np.float32).reshape(-1)  # shape=(B,)
        actions = np.asarray(predicted_y, dtype=np.float32).reshape(-1)  # shape=(B,)
        B = states.shape[0]

        x_raw_table = self.x_raw.reshape(-1)
        x_raw_query = self._norm_to_raw(states)  # shape=(B,)
        idxs = np.abs(x_raw_table[None, :] - x_raw_query[:, None]).argmin(axis=-1)  # shape=(B,)
        true_y = self.data_raw[idxs, 1]  # shape=(B,)

        errors = actions - true_y  # shape=(B,)
        rewards = - (errors ** 2)  # shape=(B,)

        next_states = np.clip(states + actions * self.dt, -1.0, 1.0)  # shape=(B,)
        dones = np.zeros(B, dtype=bool)
        self.cur_step += 1
        info = {"true_y": true_y.copy()}
        return next_states.astype(np.float32), rewards.astype(np.float32), dones, info


class Environment1DStatic:
    def __init__(self, data_raw, x_raw, state_dim, action_dim):
        assert state_dim == 1 and action_dim == 1
        self.x_raw = x_raw
        self.x_norm = norm_state(x_raw)
        self.data_raw = data_raw
        self.tolerance = 1e-3
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_step_num = 100
        self.step_count = 0
        self.state = None

    def _raw_to_norm(self, x):
        return norm_state(x)

    def _norm_to_raw(self, z):
        return denorm_state(z)

    def get_true_y(self, x_norm_scalar: float) -> float:
        x_raw = self._norm_to_raw(np.asarray(x_norm_scalar))
        idx = np.abs(self.x_raw.reshape(-1) - x_raw).argmin(axis=-1)
        return float(self.data_raw[idx, 1])

    def reset(self) -> float:
        """
        random sample
        """
        idx = np.random.randint(len(self.x_norm))
        s0 = float(self.x_norm[idx])
        self.state = s0
        self.step_count = 0
        return s0

    def step(self, predicted_y: float):
        s_scalar = float(self.state)
        a_scalar = float(predicted_y)
        true_y = self.get_true_y(s_scalar)
        error = a_scalar - true_y
        reward = - (error ** 2)
        next_s = self.reset()
        self.step_count += 1
        done = False
        if self.step_count >= self.max_step_num:
            done = True
            self.step_count = 0
        info = {"true_y": true_y, "error": error}
        return next_s, float(reward), done, info

    def batch_step(self, states_norm: np.ndarray, predicted_y: np.ndarray):
        states_norm = np.asarray(states_norm, dtype=np.float32).reshape(-1)  # shape = (B,)
        predicted_y = np.asarray(predicted_y, dtype=np.float32).reshape(-1)  # shape = (B,)
        B = states_norm.shape[0]

        x_raw_table = self.x_raw.reshape(-1)
        x_raw_query = self._norm_to_raw(states_norm)  # shape = (B,)
        idxs = np.abs(x_raw_table[None, :] - x_raw_query[:, None]).argmin(axis=-1)  # shape=(B,)
        true_y = self.data_raw[idxs, 1]  # shape=(B,)

        errors = predicted_y - true_y  # shape=(B,)
        rewards = - (errors ** 2)  # shape=(B,)

        dones = np.zeros(B, dtype=bool)
        next_states = np.random.choice(self.x_norm.reshape(-1), size=B)
        info = {"true_y": true_y, "error": errors}
        return next_states.astype(np.float32), rewards.astype(np.float32), dones, info


class Environment2DStatic:
    """
    2D surface toy environment.
    """

    def __init__(self, s_norm: np.ndarray, a_norm: np.ndarray, state_dim: int, action_dim: int, horizon: int = 100):
        assert s_norm.ndim == 2 and s_norm.shape[1] == 2
        assert a_norm.ndim == 2 and a_norm.shape[1] == 1
        assert state_dim == 2 and action_dim == 1

        self.s_norm = s_norm.astype(np.float32)  # (N, 2)
        self.a_norm = a_norm.astype(np.float32)  # (N, 1)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_states = self.s_norm.shape[0]

        self.state = None
        self.horizon = horizon
        self.cur_step = 0

    def reset(self) -> np.ndarray:
        idx = np.random.randint(0, self.num_states)
        self.state = self.s_norm[idx].copy()  # (2,)
        return self.state.copy()

    def _find_nearest_index(self, query_point: np.ndarray) -> int:
        diffs = self.s_norm - query_point[None, :]  # (N, 2)
        dists = np.linalg.norm(diffs, axis=1)  # (N,)
        return int(np.argmin(dists))

    def step(self, pred_a_norm: np.ndarray):
        a_pred = np.asarray(pred_a_norm, dtype=np.float32).reshape(-1)  # 形状 (1,)
        a_scalar = float(a_pred[0])
        s_vec = self.state.reshape(-1)  # (2,)
        idx_current = self._find_nearest_index(s_vec)
        true_a_norm = self.a_norm[idx_current]  # (1,)
        true_scalar = float(true_a_norm[0])
        error = a_scalar - true_scalar
        reward = - (error ** 2)
        next_idx = np.random.randint(0, self.num_states)
        next_state = self.s_norm[next_idx].astype(np.float32)  # (2,)
        self.state = next_state.copy()

        self.cur_step += 1
        done = False
        if self.cur_step >= self.horizon:
            done = True
        info = {"true_a_norm": true_a_norm.copy()}
        return next_state, float(reward), done, info

    def batch_step(self, states_norm: np.ndarray, pred_a_norm: np.ndarray):
        states = np.asarray(states_norm, dtype=np.float32).reshape(-1, 2)  # (B, 2)
        preds = np.asarray(pred_a_norm, dtype=np.float32).reshape(-1, 1)  # (B, 1)
        B = states.shape[0]
        diffs = states[:, None, :] - self.s_norm[None, :, :]  # (B, N, 2)
        dists = np.linalg.norm(diffs, axis=2)  # (B, N)
        idxs_current = np.argmin(dists, axis=1)  # (B,)
        true_a = self.a_norm[idxs_current]  # (B, 1)
        true_scalars = true_a.reshape(-1)  # (B,)
        errors = preds.reshape(-1) - true_scalars  # (B,)
        rewards = - (errors ** 2)  # (B,)
        next_idxs = np.random.randint(0, self.num_states, size=B)
        next_states = self.s_norm[next_idxs].astype(np.float32)  # (B, 2)
        dones = np.zeros(B, dtype=bool)
        info = {"true_a_norm": true_a.copy()}  # (B, 1)
        return next_states, rewards.astype(np.float32), dones, info


class Environment2DDynamic:
    def __init__(
            self,
            s_norm: np.ndarray,
            a_norm: np.ndarray,
            state_dim: int,
            action_dim: int,
            dt: float = 0.1,
            horizon: int = 100
    ):
        assert state_dim == 2 and action_dim == 1

        self.s_norm = s_norm.astype(np.float32)  # (N, 2)
        self.a_norm = a_norm.astype(np.float32)  # (N, 1)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.num_states = self.s_norm.shape[0]
        self.dt = dt
        self.horizon = horizon

        self.cur_step = 0
        self.state = None

    def _find_nearest_index(self, query_point: np.ndarray) -> int:
        diffs = self.s_norm - query_point[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        return int(np.argmin(dists))

    def reset(self, batch_size: int = 1) -> np.ndarray:
        self.cur_step = 0
        idxs = np.random.randint(0, self.num_states, size=batch_size)
        states = self.s_norm[idxs].astype(np.float32)  # shape = (batch_size, 2)
        if batch_size == 1:
            self.state = states[0].copy()  # (2,)
            return self.state.copy()
        else:
            self.state = states[-1].copy()
            return states

    def step(self, pred_a_norm: np.ndarray):
        a_pred = np.asarray(pred_a_norm, dtype=np.float32).reshape(-1)  # (1,)
        a_scalar = float(a_pred[0])
        s_vec = self.state.reshape(-1)  # (2,)
        idx_current = self._find_nearest_index(s_vec)
        true_a_norm = self.a_norm[idx_current]  # (1,)
        true_scalar = float(true_a_norm[0])

        error = a_scalar - true_scalar
        reward = - (error ** 2)
        next_continuous = s_vec + np.array([a_scalar * self.dt,
                                            a_scalar * self.dt], dtype=np.float32)
        next_continuous = np.clip(next_continuous, -1.0, 1.0)  # (2,)
        idx_next = self._find_nearest_index(next_continuous)
        next_state = self.s_norm[idx_next].astype(np.float32)  # (2,)
        self.cur_step += 1
        done = self.cur_step >= self.horizon
        self.state = next_state.copy()
        info = {"true_a_norm": true_a_norm.copy()}  # (1,)

        return next_state, float(reward), done, info

    def batch_step(self, states_norm: np.ndarray, pred_a_norm: np.ndarray):
        states = np.asarray(states_norm, dtype=np.float32).reshape(-1, 2)  # (B,2)
        preds = np.asarray(pred_a_norm, dtype=np.float32).reshape(-1, 1)  # (B,1)
        B = states.shape[0]

        diffs = states[:, None, :] - self.s_norm[None, :, :]  # (B,N,2)
        dists = np.linalg.norm(diffs, axis=2)  # (B,N)
        idxs_current = np.argmin(dists, axis=1)  # (B,)

        true_a = self.a_norm[idxs_current]  # (B,1)
        true_scalars = true_a.reshape(-1)  # (B,)

        errors = preds.reshape(-1) - true_scalars  # (B,)
        rewards = - (errors ** 2)  # (B,)

        move = preds.reshape(-1, 1) * self.dt  # (B,1)
        moves_2d = np.concatenate([move, move], axis=1)  # (B,2)
        next_continuous = states + moves_2d  # (B,2)
        next_continuous = np.clip(next_continuous, -1.0, 1.0)  # (B,2)

        diffs_next = next_continuous[:, None, :] - self.s_norm[None, :, :]  # (B,N,2)
        dists_next = np.linalg.norm(diffs_next, axis=2)  # (B,N)
        idxs_next = np.argmin(dists_next, axis=1)  # (B,)
        next_states = self.s_norm[idxs_next].astype(np.float32)  # (B,2)

        self.cur_step += 1
        dones = np.zeros(B, dtype=bool)
        if self.cur_step >= self.horizon:
            dones[:] = True

        info = {"true_a_norm": true_a.copy()}  # (B,1)
        return next_states, rewards.astype(np.float32), dones, info

class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)

class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """

    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

class SineEnv:
    def __init__(self, data, x_coordinates):
        #self.x_min = -10
        #self.x_max = 10
        #self.y_min = 0
        #self.y_max = 20
        self.x_min = -1
        self.x_max = 1
        self.y_min = -2
        self.y_max = 2
        self.tolerance = 1e-3  # Set the tolerance threshold
        self.data = data
        self.x_coordinates = x_coordinates

        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([10]), dtype=np.float32)

        self.max_step_num = 100
        self.step_count = 0

    def seed(self, seed=0):
        mujoco_env.MujocoEnv.seed(self, seed)

    def get_reward(self, predicted_y, true_y):
        # Calculate the reward based on the prediction error
        return -torch.abs(predicted_y - true_y)

    def reset(self):
        index = np.random.choice(self.data.shape[0], size=1)
        state_action = self.data[index]
        state, action = state_action[:,0], state_action[:,1]
        #state = torch.tensor(state).to(torch.float32).reshape(-1,1)
        #state = torch.tensor([np.random.choice(self.x_coordinates)])
        return state

    def step(self, state):
        #next_state = torch.tensor([np.random.choice(self.x_coordinates)])
        index = np.random.choice(self.data.shape[0], size=1)
        state_action = self.data[index]
        state, action = state_action[:,0], state_action[:,1]
        self.step_count += 1
        if self.step_count < self.max_step_num:
            done = False
        else:
            done = True
            self.step_count = 0
        #state = torch.tensor(state).to(torch.float32).reshape(-1,1)
        return state, 0, done, {}

    def render(self):
        pass

def get_sine_env(**kwargs):

    # Generate data from sine function
    np.random.seed(42)  # Set random seed for reproducibility
    # Define the parameters for the sine function
    amplitude = 1.0  # Amplitude of the sine wave
    frequency = 1  # Frequency of the sine wave
    phase = 0.0  # Phase shift of the sine wave
    noise_std = 0.05  # Standard deviation of the Gaussian noise
    # Generate x values
    x = np.linspace(0, 10, num=10000)
    y = amplitude * np.sin(2 * np.pi * frequency * x + phase) + np.random.normal(0, noise_std, size=len(x))
    data = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

    # Initialize the environment
    #state_dim = 1
    #action_dim = 1
    #input_size = state_dim + action_dim
    #earning_rate = 0.01
    #return SineEnv(data, x_coordinates)
    return NormalizedBoxEnv(SineEnv(data, x))

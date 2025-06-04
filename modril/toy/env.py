import numpy as np
from modril.toy.utils import norm_state, denorm_state


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
        self.state = states.copy().astype(np.float32).reshape(-1)  # self.state 保存形状 = (batch_size,)
        return self.state.copy()

    def step(self, predicted_y: np.ndarray):
        a_arr = np.asarray(predicted_y, dtype=np.float32).reshape(-1)  # 形状 = (batch_size,)
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


class Environment2D:
    """
    2D surface toy environment.
    """

    def __init__(self, s_norm: np.ndarray, a_norm: np.ndarray, state_dim: int, action_dim: int):
        assert s_norm.ndim == 2 and s_norm.shape[1] == 2
        assert a_norm.ndim == 2 and a_norm.shape[1] == 1
        assert state_dim == 2 and action_dim == 1

        self.s_norm = s_norm
        self.a_norm = a_norm
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_states = self.s_norm.shape[0]

    def reset(self) -> np.ndarray:
        idx = np.random.randint(0, self.num_states)
        return self.s_norm[idx].astype(np.float32)

    def step(self, state_norm: np.ndarray, pred_a_norm: np.ndarray):
        state_norm = np.asarray(state_norm, dtype=np.float32).reshape(-1, self.state_dim)
        pred_a_norm = np.asarray(pred_a_norm, dtype=np.float32).reshape(-1, self.action_dim)
        assert not np.any(np.isnan(state_norm)), f"step(): inputs state_norm has NaN: {state_norm}"
        assert not np.any(np.isnan(pred_a_norm)), f"step(): inputs pred_a_norm has NaN: {pred_a_norm}"
        s_vec = state_norm[0]  # shape (2,)
        a_pred = pred_a_norm[0]  # shape (1,)
        diffs = self.s_norm - s_vec  # shape (N, 2)
        dists = np.linalg.norm(diffs, axis=1)  # shape (N,)
        idx = int(np.argmin(dists))
        true_a_norm = self.a_norm[idx]  # shape (1,)
        error = a_pred - true_a_norm  # shape (1,)
        reward = - float(np.mean(error ** 2))
        next_idx = np.random.randint(0, self.num_states)
        next_state = self.s_norm[next_idx]  # shape (2,)
        next_state = next_state.astype(np.float32)  # (2,)
        true_a_norm = true_a_norm.astype(np.float32)  # (1,)
        done = False
        info = {"true_a_norm": true_a_norm.copy()}
        return next_state, reward, done, info

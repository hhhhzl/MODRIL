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

        self.s_norm = s_norm.astype(np.float32)   # (N, 2)
        self.a_norm = a_norm.astype(np.float32)   # (N, 1)
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
        states = self.s_norm[idxs].astype(np.float32)   # shape = (batch_size, 2)
        if batch_size == 1:
            self.state = states[0].copy()               # (2,)
            return self.state.copy()
        else:
            self.state = states[-1].copy()
            return states

    def step(self, pred_a_norm: np.ndarray):
        a_pred = np.asarray(pred_a_norm, dtype=np.float32).reshape(-1)   # (1,)
        a_scalar = float(a_pred[0])
        s_vec = self.state.reshape(-1)                                   # (2,)
        idx_current = self._find_nearest_index(s_vec)
        true_a_norm = self.a_norm[idx_current]                            # (1,)
        true_scalar = float(true_a_norm[0])

        error = a_scalar - true_scalar
        reward = - (error ** 2)
        next_continuous = s_vec + np.array([a_scalar * self.dt,
                                            a_scalar * self.dt], dtype=np.float32)
        next_continuous = np.clip(next_continuous, -1.0, 1.0)             # (2,)
        idx_next = self._find_nearest_index(next_continuous)
        next_state = self.s_norm[idx_next].astype(np.float32)             # (2,)
        self.cur_step += 1
        done = self.cur_step >= self.horizon
        self.state = next_state.copy()
        info = {"true_a_norm": true_a_norm.copy()}  # (1,)

        return next_state, float(reward), done, info

    def batch_step(self, states_norm: np.ndarray, pred_a_norm: np.ndarray):
        states = np.asarray(states_norm, dtype=np.float32).reshape(-1, 2)   # (B,2)
        preds = np.asarray(pred_a_norm, dtype=np.float32).reshape(-1, 1)   # (B,1)
        B = states.shape[0]

        diffs = states[:, None, :] - self.s_norm[None, :, :]               # (B,N,2)
        dists = np.linalg.norm(diffs, axis=2)                              # (B,N)
        idxs_current = np.argmin(dists, axis=1)                            # (B,)

        true_a = self.a_norm[idxs_current]                                   # (B,1)
        true_scalars = true_a.reshape(-1)                                    # (B,)

        errors = preds.reshape(-1) - true_scalars                             # (B,)
        rewards = - (errors ** 2)                                             # (B,)

        move = preds.reshape(-1, 1) * self.dt                                 # (B,1)
        moves_2d = np.concatenate([move, move], axis=1)                       # (B,2)
        next_continuous = states + moves_2d                                    # (B,2)
        next_continuous = np.clip(next_continuous, -1.0, 1.0)                  # (B,2)

        diffs_next = next_continuous[:, None, :] - self.s_norm[None, :, :]     # (B,N,2)
        dists_next = np.linalg.norm(diffs_next, axis=2)                        # (B,N)
        idxs_next = np.argmin(dists_next, axis=1)                               # (B,)
        next_states = self.s_norm[idxs_next].astype(np.float32)               # (B,2)

        self.cur_step += 1
        dones = np.zeros(B, dtype=bool)
        if self.cur_step >= self.horizon:
            dones[:] = True

        info = {"true_a_norm": true_a.copy()}  # (B,1)
        return next_states, rewards.astype(np.float32), dones, info

import numpy as np
from modril.toy.utils import norm_state, denorm_state


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


class Environment:
    def __init__(self, data_raw, x_raw, state_dim, action_dim):
        """
        data_raw : ndarray  shape (N,2) ,  [:,0] = x_raw , [:,1] = y
        x_raw    : ndarray
        """
        self.x_raw = x_raw  # [0,10]
        self.x_norm = norm_state(x_raw)  # (−π,π)
        self.data_raw = data_raw  #
        self.tolerance = 1e-3
        self.state_dim = state_dim
        self.action_dim = action_dim

    def _raw_to_norm(self, x):
        return norm_state(x)

    def _norm_to_raw(self, z):
        return denorm_state(z)

    def get_true_y_b(self, x_norm):
        x_raw_query = self._norm_to_raw(np.asarray(x_norm))
        x_raw_table = self.x_raw.reshape(-1)
        idx = np.abs(x_raw_table - x_raw_query[..., None]).argmin(axis=-1)
        return self.data_raw[idx, 1]

    def get_true_y(self, x_norm):
        x_raw = self._norm_to_raw(np.asarray(x_norm))
        idx = np.abs(self.x_raw - x_raw[..., None]).argmin(axis=-1)
        return self.data_raw[idx, 1]

    def reset(self):
        idx = np.random.randint(len(self.x_norm))
        return float(self.x_norm[idx])

    def step(self, state_norm, predicted_y):
        true_y = self.get_true_y(state_norm)
        error = predicted_y - true_y
        reward = - error ** 2
        reward = sum(reward) / len(reward)
        done = True
        next_state = self.reset()
        info = {"true_y": true_y, "error": error}
        return next_state, reward, done, info

    def batch_step(self, states_norm, predicted_y):
        """
        states_norm : ndarray shape (B, )  or (B, state_dim)
        predicted_y : ndarray shape (B, )  or (B, action_dim)
        Returns
        --------
        next_states : ndarray shape (B, )
        rewards     : ndarray shape (B, )
        dones       : ndarray shape (B, )  (all True)
        info        : dict(str -> ndarray)  (broadcast over batch)
        """
        states_norm = np.asarray(states_norm)
        predicted_y = np.asarray(predicted_y)
        true_y = self.get_true_y_b(states_norm)  # shape (B,)
        error = predicted_y - true_y[..., None] if predicted_y.ndim > 1 else predicted_y - true_y
        rewards = - np.mean(error ** 2, axis=-1)  # shape (B,)
        dones = np.ones_like(rewards, dtype=bool)
        next_states = np.random.choice(self.x_norm.reshape(-1), size=states_norm.shape[0])
        info = {"true_y": true_y, "error": error}
        return next_states, rewards, dones, info

# class Environment1:
#     """
#     把原来的“静态映射” Environment 改成“多步动态”环境
#     - data_raw, x_raw: 直接复用原来代码的输入，用于查询“真实 y”。
#     - 我们额外引入一个“动力学更新”：s_{t+1} = s_t + a_t * dt
#     - horizon: 轨迹长度
#     - reset() → 返回初始 s ∈ (-π,π)，并把 cur_step 置 0
#     - step() / batch_step() → 返回下一个状态、奖励、done=False（直到到达 horizon 步才返回 done=True）
#     """
#     def __init__(self,
#                  data_raw: np.ndarray,
#                  x_raw: np.ndarray,
#                  state_dim: int,
#                  action_dim: int,
#                  dt: float = 0.05,
#                  horizon: int = 20):
#         """
#         data_raw : ndarray, shape = (N,2), data_raw[:,0]=x_raw, data_raw[:,1]=y
#         x_raw    : ndarray,  原始 x 数值域 (e.g. [0,10])
#         state_dim, action_dim: 目前统一是 1
#         dt       : 每一步的时间步长
#         horizon  : 轨迹的最大步数
#         """
#         assert state_dim == 1 and action_dim == 1
#         self.x_raw = x_raw
#         self.x_norm = norm_state(x_raw)  # (-π, π) 之间
#         self.data_raw = data_raw
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#
#         self.dt = dt
#         self.horizon = horizon
#         self.cur_step = None
#         self.state = None
#
#     def _raw_to_norm(self, x):
#         return norm_state(x)
#
#     def _norm_to_raw(self, z):
#         return denorm_state(z)
#
#     def get_true_y_b(self, x_norm):
#         x_raw_query = self._norm_to_raw(np.asarray(x_norm))
#         x_raw_table = self.x_raw.reshape(-1)
#         idx = np.abs(x_raw_table - x_raw_query[..., None]).argmin(axis=-1)
#         return self.data_raw[idx, 1]
#
#     def get_true_y(self, x_norm):
#         x_raw = self._norm_to_raw(np.asarray(x_norm))
#         idx = np.abs(self.x_raw - x_raw[..., None]).argmin(axis=-1)
#         return self.data_raw[idx, 1]
#
#     def reset(self, batch_size: int = 1):
#         self.cur_step = 0
#         states = np.random.choice(self.x_norm.reshape(-1), size=batch_size)
#         self.state = states.copy().astype(np.float32)
#         return self.state.copy()
#
#     def step(self, state_norm: np.ndarray, predicted_y: np.ndarray):
#         s_t = np.asarray(state_norm, dtype=np.float32).reshape(-1)   # (1,)
#         a_t = np.asarray(predicted_y, dtype=np.float32).reshape(-1) # (1,)
#         true_y = self.get_true_y(s_t[0])
#         r_t = - (a_t[0] - true_y)**2
#         next_s = s_t + a_t * self.dt
#         self.cur_step += 1
#         done = False
#         if self.cur_step >= self.horizon:
#             done = True
#         self.state = next_s.copy()
#         info = {"true_y": true_y}
#         return next_s.astype(np.float32)[0], r_t, done, info
#
#     def batch_step(self, states_norm: np.ndarray, predicted_y: np.ndarray):
#         states = np.asarray(states_norm, dtype=np.float32).reshape(-1)  # shape = (B,)
#         actions = np.asarray(predicted_y, dtype=np.float32).reshape(-1) # shape = (B,)
#         B = states.shape[0]
#         true_y = self.get_true_y_b(states)
#         errors = actions - true_y
#         rewards = - (errors ** 2)
#         next_states = states + actions * self.dt
#         dones = np.zeros(B, dtype=bool)
#         info = {"true_y": true_y.copy()}
#         return next_states.astype(np.float32), rewards.astype(np.float32), dones, info

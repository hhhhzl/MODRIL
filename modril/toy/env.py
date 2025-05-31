import numpy as np
from modril.toy.utils import norm_state, denorm_state


class Environment2D:
    """
    for 2D surface toy tasks
    """

    def __init__(self, s_norm, a_norm):
        self.s_norm = s_norm  # np.ndarray shape (N,2)
        self.a_norm = a_norm  # np.ndarray shape (N,1)

    def reset(self):
        idx = np.random.randint(len(self.s_norm))
        return self.s_norm[idx]

    def step(self, state_norm, pred_a_norm):
        dists = np.linalg.norm(self.s_norm - state_norm, axis=1)
        idx = dists.argmin()
        true_a_norm = self.a_norm[idx]
        next_state = self.reset()
        return next_state, true_a_norm


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

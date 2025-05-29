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
    def __init__(self, data_raw, x_raw):
        """
        data_raw : ndarray  shape (N,2) ,  [:,0] = x_raw , [:,1] = y
        x_raw    : ndarray
        """
        self.x_raw = x_raw  # [0,10]
        self.x_norm = norm_state(x_raw)  # (−π,π)
        self.data_raw = data_raw  #
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

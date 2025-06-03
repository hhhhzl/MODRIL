import numpy as np
import matplotlib.pyplot as plt
from modril.toy.toy_tasks import Sine1D
from modril.toy.utils import norm_state, denorm_state

num_states = 30
num_actions = 30

# —— ① 从环境里取出 x_raw 和 x_norm，并排序 ——
task_tmp = Sine1D()
env_tmp = task_tmp.env

# 拿到环境内部那条“真实 x_raw 曲线” 及其归一化版本 x_norm
x_raw_env = env_tmp.x_raw  # 可能形状是 (N,1,1) 或者 (N,1)
x_norm_env = env_tmp.x_norm  # 同样 (N,1,1) 或者 (N,1)

# 先按 x_raw_env 排序，再对应地对 x_norm_env 排序
sort_idx = np.argsort(x_raw_env.reshape(-1))  # 先打平为 (N,) 再排序
x_raw_sorted = x_raw_env.reshape(-1)[sort_idx]  # 得到一维 (N,) 的原始 x 值
x_norm_sorted = x_norm_env.reshape(-1)[sort_idx]  # 得到一维 (N,) 的归一化状态

# 从 N 个点里等间距挑 30 个索引
N = len(x_raw_sorted)
indices_30 = np.linspace(0, N - 1, num_states, dtype=int)

# ② 只取这 30 个，一并 squeeze 成 (30,)
state_grid_raw = x_raw_sorted[indices_30].squeeze()  # (30,)
state_grid_norm = x_norm_sorted[indices_30].squeeze()  # (30,)

# —— action_grid 不变 ——
action_grid = np.linspace(-1, 1, num_actions)

T = 50
R_track = np.zeros((num_states, num_actions))

for i, s0_norm in enumerate(state_grid_norm):
    for j, a0 in enumerate(action_grid):
        task_n = Sine1D()
        env_local = task_n.env
        env_local.horizon = T

        # —— 直接把环境的归一化状态设为 s0_norm ——
        env_local.state = np.array([s0_norm], dtype=np.float32)
        env_local.cur_step = 0

        cumulative_r = 0.0

        # —— 第一步，用候选 a0 ——
        a0_arr = np.array([a0], dtype=np.float32)
        next_s_arr, r_arr, done, _ = env_local.step(a0_arr)
        cumulative_r += float(r_arr)
        s_arr = next_s_arr

        # —— 后面每一步，用环境真正认可的 x_t → sin(x_t) 作为 expert ——
        for t in range(1, T):
            s_t_norm = float(s_arr)
            x_t_env = denorm_state(s_t_norm)
            a_t_true = np.sin(x_t_env)
            a_t_arr = np.array([a_t_true], dtype=np.float32)

            next_s_arr, r_arr, done, _ = env_local.step(a_t_arr)
            cumulative_r += float(r_arr)
            s_arr = next_s_arr
            if done:
                break

        R_track[i, j] = cumulative_r

# —— 画热力图 ——
plt.figure(figsize=(6, 5))
plt.imshow(
    R_track,
    origin='lower',
    aspect='auto',
    extent=[action_grid.min(), action_grid.max(),
            state_grid_raw.min(), state_grid_raw.max()],
    cmap='coolwarm'
)
plt.colorbar(label='Cumulative Env Reward')
plt.xlabel('Initial action a0 (≈ sin(x0_raw))')
plt.ylabel('Initial raw x0 ∈ [−3,3]')
plt.title(f'Sine Rollout Reward Heatmap (T={T})')
plt.show()

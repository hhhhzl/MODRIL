import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from modril.modril.model_base_diffusion import MBDScore
from modril.toy.toy_tasks import Sine1D

# ——— 步骤 1：生成一批专家样本，用来计算 g_E_mean ———
N_expert = 200
expert_batch_s = np.random.uniform(-3.0, 3.0, size=(N_expert,))
expert_batch_a = np.sin(expert_batch_s)

# ——— 步骤 2：初始化 MBDScore，开启推荐参数（toy 环境里必须这样） ———
task = Sine1D()
mbd = MBDScore(
    task.env,
    env_name="toy",
    steps=200,
    device="cpu",
    disable_recommended_params=False
    )

# ——— 步骤 3：计算专家平均 g_E_mean ———
gE_vals = mbd.compute_reward(expert_batch_s.astype(np.float32), expert_batch_a.astype(np.float32))  # shape=(200,)
gE_mean = gE_vals.mean()  # scalar

# ——— 步骤 4：为绘制 Surface，先准备一个“action 网格” 和 “state 网格” ———
num_states = 50
num_actions = 50
state_grid = np.linspace(-3.0, 3.0, num_states)  # 比如 50 个 state
action_grid = np.linspace(-2.0, 2.0, num_actions)  # 自行选择 action 范围（-2..2）

# ——— 步骤 5：对所有 (state, action) 网格点，都调用一次 compute_reward ———
data = []
for s in state_grid:
    for a in action_grid:
        # 注意：compute_reward 的输入都要是 1D float32 array
        gA = mbd.compute_reward(np.array([s], dtype=np.float32),
                                np.array([a], dtype=np.float32))  # shape=(1,)
        r = (gE_mean - gA[0])
        data.append((s, a, r))

df = pd.DataFrame(data, columns=['state', 'action', 'reward'])

# ——— 步骤 6：用 pivot 得到完整矩阵——此时绝不会剩下 NaN ———
pivot_table = df.pivot(index='state', columns='action', values='reward')
state_vals = pivot_table.index.values  # shape=(num_states,)
action_vals = pivot_table.columns.values  # shape=(num_actions,)
reward_vals = pivot_table.values  # shape=(num_states, num_actions)

# ——— 步骤 7：画 3D Surface ———
X, Y = np.meshgrid(action_vals, state_vals)  # X.shape=(num_states, num_actions)
Z = reward_vals  # Z.shape=(num_states, num_actions)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('Action')
ax.set_ylabel('State')
ax.set_zlabel('Reward')
ax.set_title('Reward Surface')
plt.show()

# ——— 步骤 8：画 2D 热力图（imshow） ———
plt.figure(figsize=(8, 6))
plt.imshow(
    reward_vals,
    origin='lower',
    aspect='auto',
    extent=[action_vals.min(), action_vals.max(),
            state_vals.min(), state_vals.max()],
    cmap='coolwarm'
)
plt.colorbar(label='Reward')
plt.xlabel('Action')
plt.ylabel('State')
plt.title('Reward Heatmap')
plt.show()

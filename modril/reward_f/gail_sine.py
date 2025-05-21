import gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import wandb
import datetime
from modril.reward_f.env import Environment
from modril.reward_f.utils import compute_advantage


def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def norm_state(x):
    return (x - 5.0) / 5.0 * np.pi

def denorm_state(z):
    return z * 5.0 / np.pi + 5.0


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 直接存 action_dim 个参数

    def forward(self, x):  # (B, state_dim)
        if x.dim() == 1:  # (state_dim,)
            x = x.unsqueeze(0)
        h = self.backbone(x)
        mean = self.fc_mean(h)  # (B, action_dim)
        log_std = torch.clamp(self.log_std, -4, 2).expand_as(mean)
        return mean, log_std


# ========= 两层 MLP Critic =========
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  # (B, state_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.v_net(x)  # (B,1)


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.actor = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mean, log_std = self.actor(state)

        log_std = log_std.clamp(-4, 1)  # std ∈ [e^-4, e^1≈2.7]
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).unsqueeze(1).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        advantage = torch.nan_to_num(advantage, nan=0.0, posinf=1e4, neginf=-1e4)
        advantage = torch.clamp(advantage, -1e4, 1e4)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        with torch.no_grad():
            mean, log_std = self.actor(states)
            dist = torch.distributions.Normal(mean, log_std.exp())
            old_log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True).detach()

        for _ in range(self.epochs):
            mean, log_std = self.actor(states)
            dist = torch.distributions.Normal(mean, log_std.exp())
            log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        s, a = s.unsqueeze(-1) if s.dim() == 1 else s, a.unsqueeze(-1) if a.dim() == 1 else a
        x = torch.cat([s, a], 1)
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))


class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d):
        self.discriminator = Discriminator(state_dim, hidden_dim, action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
        self.agent = agent

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s):
        expert_states = torch.tensor(expert_s, dtype=torch.float).view(-1, 1).to(device)
        expert_actions = torch.tensor(expert_a, dtype=torch.float).view(-1, 1).to(device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).view(-1, 1).to(device)
        agent_actions = torch.tensor(agent_a, dtype=torch.float).view(-1, 1).to(device)
        # expert_actions = F.one_hot(expert_actions.to(torch.int64), num_classes=2).float()
        # agent_actions = F.one_hot(agent_actions.to(torch.int64), num_classes=2).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        # print("D_expert", expert_prob.mean(), "D_agent", agent_prob.mean())
        discriminator_loss = F.binary_cross_entropy(expert_prob, torch.ones_like(expert_prob)) + \
                             F.binary_cross_entropy(agent_prob, torch.zeros_like(agent_prob))
        # print("d_loss", discriminator_loss)
        # discriminator_loss = F.binary_cross_entropy(agent_prob, torch.ones_like(agent_prob)) + F.binary_cross_entropy(expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        # wandb.log({'discriminator_loss':discriminator_loss})

        with torch.no_grad():
            rewards = -torch.log(1 - agent_prob + 1e-8).cpu().numpy()
        # wandb.log({'agent_prob':agent_prob[0]})
        # wandb.log({'reward for policy training':rewards[0]})
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': [False] * len(agent_s)
        }
        self.agent.update(transition_dict)


current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# Generate data from sine function
np.random.seed(42)  # Set random seed for reproducibility
# Define the parameters for the sine function
amplitude = 1  # Amplitude of the sine wave
frequency = 0.1  # Frequency of the sine wave
phase = 0.0  # Phase shift of the sine wave
noise_std = 0.05  # Standard deviation of the Gaussian noise
scale = 2
# Generate x values
x = np.linspace(0, 10, num=1000)
y = amplitude * np.sin(scale * frequency * np.pi * x + phase) + np.random.normal(0, noise_std, size=len(x))
data_raw = np.stack([x, y], axis=1)

expert_s = norm_state(x)
expert_a = y
data = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

# Initialize the environment
state_dim = 1
action_dim = 1
input_size = state_dim + action_dim
learning_rate = 0.01
env = Environment(data_raw, x)

# Initialize the RL agent
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 250
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

lr_d = 1e-3
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
gail = GAIL(agent, state_dim, action_dim, hidden_dim, lr_d)
gail.discriminator.apply(reset_weights)
n_episode = 500
return_list = []

with tqdm(total=n_episode, desc="Progress") as pbar:
    for i in range(n_episode):
        state = env.reset()
        state_list = []
        action_list = []
        next_state_list = []
        episode_returns = []
        # while not done:
        for i in range(100):
            action = agent.take_action(state)
            next_state, true_y = env.step(state, action)
            episode_returns.append(abs(action - true_y))
            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
            state = next_state
        return_list.append(np.mean(episode_returns))
        gail.learn(expert_s, expert_a, state_list, action_list, next_state_list)
        pbar.update(1)

# Plotting the results
# plt.figure()
# plt.scatter(x, y, label='Origin Sin')
plt.scatter(expert_s, expert_a, label='Ground Truth')
plt.scatter(state_list, action_list, label='Predicted')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('result_sine.png')
plt.show()

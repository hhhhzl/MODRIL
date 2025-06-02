import torch
from modril.toy.networks import Actor, Critic
from modril.toy.utils import compute_advantage
import torch.nn.functional as F
import numpy as np

class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
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
        if isinstance(state, np.ndarray):
            state_np = state.astype(np.float32)
        else:
            state_np = np.array(state, dtype=np.float32)

        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)  # shape = (1, state_dim)
        mean, log_std = self.actor(state_tensor)
        log_std = log_std.clamp(-4, 1)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample().cpu().numpy().flatten()
        if action.shape == ():
            return float(action)
        else:
            return action

    def update(self, transition_dict):
        states_np = np.array(transition_dict['states'], dtype=np.float32)  # (batch, state_dim)
        next_states_np = np.array(transition_dict['next_states'], dtype=np.float32)
        states = torch.from_numpy(states_np).to(self.device)  # (batch, state_dim)
        next_states = torch.from_numpy(next_states_np).to(self.device)  # (batch, state_dim)
        actions_np = np.array(transition_dict['actions'])  # (batch,)
        rewards_np = np.array(transition_dict['rewards'], dtype=np.float32)  # (batch,)
        dones_np = np.array(transition_dict['dones'], dtype=np.float32)  # (batch,)
        actions = torch.from_numpy(actions_np).view(-1, 1).to(self.device)  # (batch, 1)
        rewards = torch.from_numpy(rewards_np).view(-1, 1).to(self.device)  # (batch, 1)
        dones = torch.from_numpy(dones_np).view(-1, 1).to(self.device)  # (batch, 1)

        with torch.no_grad():
            next_values = self.critic(next_states)  # (batch, 1)
            td_target = rewards + self.gamma * next_values * (1.0 - dones)  # (batch, 1)

        values = self.critic(states)  # (batch, 1)
        td_delta = td_target - values  # (batch, 1)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)  # (batch, 1)
        advantage = torch.nan_to_num(advantage, nan=0.0, posinf=1e4, neginf=-1e4)
        advantage = torch.clamp(advantage, -1e4, 1e4)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        with torch.no_grad():
            mean_old, log_std_old = self.actor(states)  # (batch, action_dim)
            std_old = log_std_old.exp()
            dist_old = torch.distributions.Normal(mean_old, std_old)
            old_log_probs = dist_old.log_prob(actions).sum(dim=1, keepdim=True)  # (batch, 1)

        for _ in range(self.epochs):
            mean, log_std = self.actor(states)  # (batch, action_dim)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)  # (batch, 1)
            ratio = torch.exp(log_probs - old_log_probs)  # (batch, 1)
            surr1 = ratio * advantage  # (batch, 1)
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # (batch, 1)
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            value_preds = self.critic(states)  # (batch, 1)
            critic_loss = F.mse_loss(value_preds, td_target.detach())  # (batch, 1) vs (batch,1)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

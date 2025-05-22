import torch
from modril.reward_f.networks import Actor, Critic
from modril.reward_f.utils import compute_advantage
import torch.nn.functional as F


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

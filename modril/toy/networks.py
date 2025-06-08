import torch
import torch.nn as nn


class SharedVNet(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(s_dim + a_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_c = nn.Linear(hidden, a_dim)
        self.head_r = nn.Linear(hidden, a_dim)

    def forward(self, a, s, t):
        h = self.shared(torch.cat([a, s, t], 1))
        return self.head_c(h), torch.tanh(self.head_r(h))


class ConditionalVNet(nn.Module):
    """
    vθ(x,t) —— time-conditioned vector field
    for flow matching
    """

    def __init__(self, s_dim, a_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(a_dim + s_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, a_dim)
        )

    def forward(self, a_t, s, t):
        # a_t: [B,a_dim], s: [B,s_dim], t: [B,1]
        x = torch.cat([a_t, s, t], dim=1)
        return self.net(x)


class VNet(nn.Module):
    """
    vθ(x,t) —— time-conditioned vector field
    for flow matching
    """

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, t):
        t_vec = t * torch.ones_like(x[:, :1])
        return self.net(torch.cat([x, t_vec], dim=1))


class ODEF(nn.Module):
    """
    CNF ODE func
    for FFJORD
    """

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim)
        )

    def forward(self, t, h):
        t_vec = t * torch.ones_like(h[:, :1])
        return self.net(torch.cat([h, t_vec], dim=1))

    # def forward(self, t, h):
    #     t_vec = torch.full_like(h[:, :1], t)
    #     return self.net(torch.cat([h, t_vec], dim=1))


class TNet(nn.Module):
    """Use for MINE"""

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s, a):
        if s.dim() == 1:
            s, a = s.unsqueeze(0), a.unsqueeze(0)
        x = torch.cat([s, a], dim=1)
        return self.net(x)


class EpsNet(nn.Module):
    """Use for diffusion Gail discriminator"""

    def __init__(self, x_dim, t_dim, label_dim, hidden=128):
        super().__init__()
        in_dim = x_dim + t_dim + label_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, x_dim)
        )

    def forward(self, x, t_emb, c_emb):
        h = torch.cat([x, t_emb, c_emb], dim=-1)
        return self.net(h)


class Actor(nn.Module):
    """For PPO"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):  # (B, state_dim)
        if x.dim() == 1:  # (state_dim,)
            x = x.unsqueeze(0)
        h = self.backbone(x)
        mean = self.fc_mean(h)  # (B, action_dim)
        log_std = torch.clamp(self.log_std, -4, 2).expand_as(mean)
        return mean, log_std


class Critic(nn.Module):
    """For PPO"""

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

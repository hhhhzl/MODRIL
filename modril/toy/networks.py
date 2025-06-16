import torch
import torch.nn as nn
import math


class GradNorm2D(nn.Module):
    def __init__(
            self,
            beta_init=1e-4,
            gamma_init=1e-5,
            alpha=0.1,
            eps=1e-8,
            beta_min=1e-6,
            beta_max=1e1,
            gamma_min=1e-8,
            gamma_max=1e-2,
            warmup_steps=100,
            update_freq=10,
            ema_decay=0.9,
            enable_beta=True,
            enable_gamma=True
    ):
        super().__init__()
        self.alpha, self.eps = alpha, eps
        self.beta_min, self.beta_max = beta_min, beta_max
        self.gamma_min, self.gamma_max = gamma_min, gamma_max
        self.warmup_steps = warmup_steps
        self.update_freq = update_freq
        self.ema_decay = ema_decay

        self.enable_beta = enable_beta
        self.enable_gamma = enable_gamma

        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta_init)))
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma_init)))

        self.register_buffer("step_count", torch.tensor(0))
        self.register_buffer("L0_anti", torch.tensor(0.))
        self.register_buffer("L0_jpen", torch.tensor(0.))
        self.register_buffer("ema_g_anti", torch.tensor(0.))
        self.register_buffer("ema_g_jpen", torch.tensor(0.))

    def forward(self, loss_anti, j_pen):
        if self.step_count < self.warmup_steps:
            self.L0_anti += loss_anti.detach()
            self.L0_jpen += j_pen.detach()
            if self.step_count == self.warmup_steps - 1:
                self.L0_anti /= float(self.warmup_steps)
                self.L0_jpen /= float(self.warmup_steps)

        self.step_count += 1
        beta = self.log_beta.exp() if self.enable_beta else torch.tensor(0.0, device=loss_anti.device)
        gamma = self.log_gamma.exp() if self.enable_gamma else torch.tensor(0.0, device=j_pen.device)
        return beta * loss_anti + gamma * j_pen, beta, gamma

    @torch.no_grad()
    def update(self, loss_anti, j_pen, params):
        if (self.step_count <= self.warmup_steps or
                self.step_count % self.update_freq != 0):
            return

        g_anti = g_jpen = 0.0
        if self.enable_beta:
            grads_anti = torch.autograd.grad(
                loss_anti, params, retain_graph=True, allow_unused=True
            )
            g_anti = sum([g.norm().item() for g in grads_anti if g is not None])
        if self.enable_gamma:
            grads_jpen = torch.autograd.grad(
                j_pen, params, retain_graph=True, allow_unused=True
            )
            g_jpen = sum([g.norm().item() for g in grads_jpen if g is not None])

        # EMA Smooth
        self.ema_g_anti = self.ema_decay * self.ema_g_anti + (1 - self.ema_decay) * g_anti
        self.ema_g_jpen = self.ema_decay * self.ema_g_jpen + (1 - self.ema_decay) * g_jpen
        g_anti, g_jpen = float(self.ema_g_anti), float(self.ema_g_jpen)

        r_anti = (loss_anti.detach() / (self.L0_anti + self.eps)).item()
        r_jpen = (j_pen.detach() / (self.L0_jpen + self.eps)).item()
        r_bar = 0.5 * (r_anti + r_jpen) + self.eps

        # GradNorm
        inv_mean_g = 0.5 * (g_anti + g_jpen) + self.eps
        if self.enable_beta:
            delta_b = self.alpha * (g_anti / inv_mean_g * (r_anti / r_bar) - 1)
            self.log_beta.data += delta_b
            self.log_beta.data.clamp_(math.log(self.beta_min), math.log(self.beta_max))
        if self.enable_gamma:
            delta_g = self.alpha * (g_jpen / inv_mean_g * (r_jpen / r_bar) - 1)
            self.log_gamma.data += delta_g
            self.log_gamma.data.clamp_(math.log(self.gamma_min), math.log(self.gamma_max))


class SharedVNet(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=256):
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

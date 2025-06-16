import torch, torch.nn as nn
import math


class SharedVNet(nn.Module):
    """
    Shared V-Net with configurable depth for both vc and residual
    """

    def __init__(self, s_dim, a_dim, hidden=256, depth=4):
        super().__init__()
        self.depth = max(1, depth)
        layers = []
        layers.append(nn.Linear(s_dim + a_dim + 1, hidden))
        layers.append(nn.ReLU())
        for _ in range(self.depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        self.shared = nn.Sequential(*layers)
        self.head_c = nn.Linear(hidden, a_dim)
        self.head_r = nn.Linear(hidden, a_dim)

    def forward(self, a, s, t):
        h = self.shared(torch.cat([a, s, t], dim=1))
        return self.head_c(h), torch.tanh(self.head_r(h))


class ConditionalVNet(nn.Module):
    """
    vθ(x,t) —— time-conditioned vector field
    for flow matching
    """

    def __init__(self, s_dim, a_dim, hidden=128, depth=4):
        super().__init__()
        self.depth = max(1, depth)
        layers = []
        layers.append(nn.Linear(a_dim + s_dim + 1, hidden))
        layers.append(nn.ReLU())
        for _ in range(self.depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, a_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, a_t, s, t):
        # a_t: [B,a_dim], s: [B,s_dim], t: [B,1]
        x = torch.cat([a_t, s, t], dim=1)
        return self.net(x)


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

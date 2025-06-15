import torch, torch.nn as nn


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

import torch, torch.nn as nn


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

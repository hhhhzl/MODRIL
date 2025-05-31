import torch
import torch.nn as nn
from modril.toy.networks import EpsNet, TNet, ODEF, VNet, ConditionalVNet
from modril.toy.utils import timestep_embed
import torch.nn.functional as F
from torchdiffeq import odeint


class Discriminator(nn.Module):
    def __init__(
            self,
            mode,
            state_dim=None,
            action_dim=None,
            hidden_dim=None,

            label_dim=10,
            T=200,
            beta_start=1e-4,
            beta_end=0.02,
            n_repeat=6
    ):
        super().__init__()
        self.mode = mode
        self.n_repeat = n_repeat

        if mode == 'gail':
            assert all(
                [v is not None for v in [state_dim, action_dim, hidden_dim]]), "Missing parameters for standard mode"
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            self.x_dim = state_dim + action_dim
            self.fc1 = nn.Linear(self.x_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)

        elif mode == 'mfd':
            assert all([v is not None for v in [state_dim, action_dim]]), "Missing parameters for Diffusion mode"
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.x_dim = self.state_dim + self.action_dim
            self.label_dim = label_dim
            self.T = T
            self.register_buffer("betas", torch.linspace(beta_start, beta_end, T))
            alphas = 1. - self.betas
            self.register_buffer("alphas_bar", torch.cumprod(alphas, dim=0))
            self.t_dim = 32
            self.net = EpsNet(self.x_dim, self.t_dim, label_dim, hidden=128)
            self.register_buffer("c_pos", F.one_hot(torch.tensor(0), label_dim).float())
            self.register_buffer("c_neg", F.one_hot(torch.tensor(1), label_dim).float())

        elif mode == 'mbd':
            pass
        elif mode == 'mine':
            pass
        elif mode == 'njw':
            pass
        elif mode == 'fford':
            pass
        elif mode == 'fm':
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def forward(self, *inputs, **kwargs):
        if self.mode == 'gail':
            s, a = inputs
            s = s.unsqueeze(-1) if s.dim() == 1 else s
            a = a.unsqueeze(-1) if a.dim() == 1 else a
            x = torch.cat([s, a], dim=1)
            x = F.relu(self.fc1(x))
            return torch.sigmoid(self.fc2(x))
        elif self.mode == 'mfd':
            x0 = inputs[0]
            n_repeat = kwargs.get('n_repeat', self.n_repeat)
            Ds = []
            for _ in range(n_repeat):
                lp, _ = self._mfd_Ldiff(x0, self.c_pos)
                ln, _ = self._mfd_Ldiff(x0, self.c_neg)
                Ds.append(torch.sigmoid(ln - lp))
            return torch.stack(Ds, dim=0).mean(dim=0)
        else:
            raise NotImplementedError

    # ===================================
    #       model-free diffusion(mfd)
    # ===================================
    def _mfd_q_sample(self, x0, t, eps):
        """forward diffusion"""
        a_bar = self.alphas_bar[t].view(-1, 1)
        return (a_bar.sqrt() * x0) + ((1 - a_bar).sqrt() * eps)

    def _mfd_Ldiff(self, x0, c_emb):
        """diffusion loss"""
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        x_t = self._mfd_q_sample(x0, t, eps)
        t_emb = timestep_embed(t.float(), self.t_dim)
        eps_pred = self.net(x_t, t_emb, c_emb.expand(B, -1))
        return F.mse_loss(eps_pred, eps, reduction='none').mean(dim=1), t

    @torch.no_grad()
    def get_reward(self, x0, eps=1e-8):
        """calculate reward"""
        if self.mode != 'mfd':
            raise NotImplementedError("get_reward is only available in diffusion mode")
        D = self.forward(x0).clamp(eps, 1 - eps)
        return torch.log(D) - torch.log(1 - D)


class MI_Estimator:
    """
    Support 'mine' and 'nwj'
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            device,
            lr=1e-4,
            hidden=128,
            mode="mine",
            ma_rate=0.01
    ):
        assert mode in {"mine", "nwj"}
        self.mode, self.device = mode, device
        self.T = TNet(state_dim, action_dim, hidden).to(device)
        self.opt = torch.optim.Adam(self.T.parameters(), lr=lr)
        self.ma_et = None
        self.ma_rate = ma_rate

    def estimate_and_update(self, s_E, a_E, s_A, a_A):
        """"""
        s_E = torch.tensor(s_E, dtype=torch.float32, device=self.device)
        a_E = torch.tensor(a_E, dtype=torch.float32, device=self.device)
        s_A = torch.tensor(s_A, dtype=torch.float32, device=self.device)
        a_A = torch.tensor(a_A, dtype=torch.float32, device=self.device)

        # if they came in as 1-D, make them (batch,1)
        if s_E.dim() == 1:
            s_E = s_E.unsqueeze(1)

        if a_E.dim() == 1:
            a_E = a_E.unsqueeze(1)

        if s_A.dim() == 1:
            s_A = s_A.unsqueeze(1)

        if a_A.dim() == 1:
            a_A = a_A.unsqueeze(1)

        T_E = self.T(s_E, a_E)
        T_A = self.T(s_A, a_A)

        if self.mode == "mine":
            # log E_pi[exp(T)]
            et = torch.exp(T_A)
            mean_et = et.mean()
            # moving average trick
            if self.ma_et is None:
                self.ma_et = mean_et.detach()
            else:
                self.ma_et += self.ma_rate * (mean_et.detach() - self.ma_et)
            mine_loss = -(T_E.mean() - torch.log(mean_et) * mean_et.detach() / self.ma_et)
            loss = mine_loss
        else:  # NWJ
            nwj_loss = -(T_E.mean() - (torch.exp(T_A - 1).mean()))
            loss = nwj_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return T_A.detach().cpu().numpy().squeeze()


# ---------------- 3. FFJORD Density ----------------
class FFJORDDensity(nn.Module):
    """
    log p(x) = log p(z_T) − ∫_0^T Tr(∂f/∂z_t) dt
    """

    def __init__(self, dim, T=1.0, hidden=64):
        super().__init__()
        self.T = T
        self.func = ODEF(dim, hidden)
        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(dim), torch.eye(dim)
        )

    def _divergence_approx(self, y, f):
        """Tr(∂f/∂y) ≈ vᵀJᵀv,  v~Rademacher"""
        v = torch.randint_like(y, low=0, high=2).float() * 2 - 1  # ±1
        (Jv,) = torch.autograd.grad(f, y, v, create_graph=True)
        return (Jv * v).sum(1)

    def _odefunc(self, t, states):
        z, logp = states
        z.requires_grad_(True)
        f = self.func(t, z)
        div_f = self._divergence_approx(z, f).unsqueeze(1)
        return f, -div_f

    # ---------- Compute log probability ----------
    @torch.no_grad()
    def log_prob(self, x, atol=1e-5, rtol=1e-5):
        """
        get log prob
        """
        with torch.enable_grad():
            z0 = x.requires_grad_(True)
            # z0 = x.detach().requires_grad_(True)
            logp0 = torch.zeros(x.size(0), 1, device=x.device)
            t_span = torch.tensor([0., self.T], device=x.device)
            zT, logpT = odeint(self._odefunc, (z0, logp0), t_span, atol=atol, rtol=rtol)
        zT, logpT = zT[-1], logpT[-1].squeeze(1)
        return self.prior.log_prob(zT) + logpT  # (B,)

    # ---------- Training loss (NLL) ----------
    def nll(self, x, atol=1e-5, rtol=1e-5):
        z0 = x.requires_grad_(True)
        # z0 = x.detach().requires_grad_(True)
        logp0 = torch.zeros(x.size(0), 1, device=x.device)
        t_span = torch.tensor([0., self.T], device=x.device)
        zT, logpT = odeint(self._odefunc, (z0, logp0), t_span, atol=atol, rtol=rtol)
        zT, logpT = zT[-1], logpT[-1].squeeze(1)
        logp = self.prior.log_prob(zT) + logpT
        return -logp.mean()


class FlowMatching(nn.Module):
    """
    Ho & Salimans 2023: flow matching
    goal:  E_{t∼U(0,1)} [ || vθ(x_t,t) - v∗(x_t,t) ||² ]
    x_t = (1-t)·x + t·x̃  ，x̃~N(0,I)
    """

    def __init__(self, s_dim, a_dim, device, eps=1e-2):
        super().__init__()
        self.s_dim, self.a_dim = s_dim, a_dim
        # self.vnet = VNet(dim)
        self.vnet = ConditionalVNet(s_dim, a_dim).to(device)
        self.dim = self.s_dim + self.a_dim
        # self.prior = torch.distributions.MultivariateNormal(
        #     torch.zeros(dim), torch.eye(dim)
        # )
        self.prior = torch.distributions.Normal(0, 1)
        self.eps = eps

    def _v_star(self, x_t, x0, t):
        return (x0 - self.prior.mean.to(x0)) / (1 - t)

    def fm_loss(self, x0):
        B = x0.size(0)
        t = torch.rand(B, 1, device=x0.device) * (1 - 2 * self.eps) + self.eps
        noise = self.prior.sample((B,)).to(x0)
        x_t = (1 - t) * x0 + t * noise

        v_star = self._v_star(x_t, x0, t)  # [B, D]
        v_pred = self.vnet(x_t, t)  # [B, D]

        weight = (1 - t).pow(2)  # [B,1]
        mse_per_sample = F.mse_loss(v_pred, v_star, reduction='none').sum(dim=1, keepdim=True)  # [B,1]
        loss = (weight * mse_per_sample).mean()
        return loss

    def c_fm_loss(self, s, a0, dequant_std=0.02):
        """
        """
        B = s.size(0)
        a0 = a0 + torch.randn_like(a0) * dequant_std
        t = torch.rand(B, 1, device=a0.device) * (1 - 2 * self.eps) + self.eps

        noise = self.prior.sample((B, self.a_dim)).to(a0)
        a_t = (1 - t) * a0 + t * noise
        v_star = (a0 - 0.) / (1 - t)  # prior.mean = 0

        v_pred = self.vnet(a_t, s, t)
        weight = (1 - t).pow(2)  # [B,1]
        mse_i = ((v_pred - v_star) ** 2).sum(dim=1, keepdim=True)  # [B,1]
        return (weight * mse_i).mean()

    # ------ estimate log-ratio via path integral ------
    def log_prob(self, x, n_steps=16):
        """
        conditional log_prob
        """
        B = x.size(0)
        s = x[:, :self.s_dim]  # [B, s_dim]
        a = x[:, self.s_dim:]  # [B, a_dim]

        t_grid = torch.linspace(0, 1, n_steps + 1, device=x.device)  # [n_steps+1]
        a_t = a
        sum_trap = torch.zeros(B, device=x.device)
        for i in range(n_steps):
            t_mid = 0.5 * (t_grid[i] + t_grid[i + 1])  # scalar
            t_mid_batch = t_mid.expand(B, 1)  # [B,1]
            v = self.vnet(a_t, s, t_mid_batch)  # [B, a_dim]

            # step = Δt = 1 / n_steps
            delta = 1.0 / n_steps

            # Euler
            a_t = a_t + v * delta  # [B, a_dim]
            sum_trap += v.norm(dim=1) * delta  # [B]

        logp_prior = self.prior.log_prob(a_t)  #
        logp_prior = logp_prior.sum(dim=1)  # [B]

        # 4) return log p = log p_prior - ∫||v||dt
        return logp_prior - sum_trap  # [B]

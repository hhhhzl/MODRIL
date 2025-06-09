import math, torch, torch.nn as nn, torch.nn.functional as F
from modril.toy.utils import dynamic_convert
from modril.toy.networks import ConditionalVNet, SharedVNet
from torch.autograd import grad

_RNG = torch.Generator()


def _rademacher(shape_or_tensor, device=None):
    """Generate Rademacher (±1) noise matching *shape* or *tensor*."""
    # --- accept Tensor or torch.Size/tuple ---
    if isinstance(shape_or_tensor, torch.Tensor):
        device = shape_or_tensor.device if device is None else device
        shape = tuple(shape_or_tensor.shape)
    else:  # shape tuple / torch.Size
        shape = shape_or_tensor
        assert device is not None, "device must be specified when passing shape"

    return torch.randint(0, 2, shape, device=device, generator=_RNG).float().mul_(2).sub_(1)


def _hutchinson_div(y: torch.Tensor, f: torch.Tensor, k: int = 1) -> torch.Tensor:
    """ Hutchinson-trace estimator ∇·f with k Rademacher vectors """
    B, D = y.shape
    out = 0.0
    for _ in range(k):
        v = _rademacher((B, D), device=y.device)
        (jv,) = torch.autograd.grad(
            outputs=f,
            inputs=y,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
        )
        out = out + (jv * v).sum(dim=-1)
    return out / k


# def _jacobian_frobenius(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
#     B, D = x.shape
#     x = x.detach().requires_grad_(True)
#     norms = torch.zeros(B, device=x.device)
#     for i in range(D):
#         grads = grad(
#             outputs=f[:, i].sum(),
#             inputs=x,
#             create_graph=True,
#             retain_graph=True,
#             allow_unused=True
#         )
#         gi = grads[0]
#         if gi is not None:
#             norms += gi.pow(2).sum(dim=1)
#     return norms

# def _hutchinson_div(y: torch.Tensor, f: torch.Tensor, k: int = 1) -> torch.Tensor:
#     B, D = y.shape
#     v = _rademacher((B, D), device=y.device)
#     (jv,) = torch.autograd.grad(
#         outputs=f,
#         inputs=y,
#         grad_outputs=v,
#         create_graph=True,
#         retain_graph=True,
#         allow_unused=True
#     )
#     if jv is None:
#         return torch.zeros(B, device=y.device)
#     return (jv * v).sum(dim=-1)

def _jacobian_frobenius(x: torch.Tensor, f: torch.Tensor):
    """‖∇_x f‖_F² via one Hutchinson vector (cheapest)."""
    v = _rademacher(x, x.device)
    (jv,) = torch.autograd.grad(f, x, v, create_graph=True, retain_graph=True, only_inputs=True)
    return (jv.pow(2)).sum(-1)  # [B]


class CoupledFlowMatching(nn.Module):
    """v_E = v_c + r_φ,
    v_π = v_c − r_φ
    (anti‑symmetric residual).
    """

    def __init__(self, s_dim: int, a_dim: int, eps: float = 1e-3):
        super().__init__()
        self.s_dim, self.a_dim, self.eps = s_dim, a_dim, eps
        self.net = SharedVNet(s_dim, a_dim, hidden=128)
        self.prior = torch.distributions.Normal(0., 1.)

    def v_field(self, a_t: torch.Tensor, s: torch.Tensor, t: torch.Tensor, role: str):
        v_c, r = self.net(a_t, s, t)
        return v_c + r if role == "expert" else v_c - r

    @staticmethod
    def _v_star(a_t, a0, t, noise):
        return noise - a0

    def c_fm_loss(self, s, a0, role):
        B = a0.size(0)
        device = a0.device
        t = torch.rand(B, 1, device=device) * (1 - 2 * self.eps) + self.eps
        noise = self.prior.sample((B, self.a_dim)).to(device)
        a_t = (1 - t) * a0 + t * noise
        v_star = self._v_star(a_t, a0, t, noise)
        v_pred = self.v_field(a_t, s, t, role)
        w = (1 - t).pow(2)
        return (w * F.mse_loss(v_pred, v_star, reduction="none").sum(1, keepdim=True)).mean()

    def log_prob(self, s: torch.Tensor, a0: torch.Tensor, role: str, k_max: int = 1, tau: float = 0.5, n_steps: int = 32
                 ) -> torch.Tensor:
        B = a0.size(0)
        device = a0.device
        t_grid = torch.linspace(0., 1., n_steps + 1, device=device)
        delta = 1.0 / n_steps

        log_det = torch.zeros(B, device=device)
        a_t = a0.detach()

        for k in range(n_steps):
            t_mid = 0.5 * (t_grid[k] + t_grid[k + 1])
            a_t = a_t.detach().requires_grad_(True)
            v_t = self.v_field(a_t, s, t_mid.expand(B, 1), role)
            div = _hutchinson_div(a_t, v_t)
            log_det -= delta * div
            a_t = a_t + v_t.detach() * delta

        logp_prior = self.prior.log_prob(a_t).sum(-1)
        return logp_prior + log_det


################################################################################
# Learner                                                                      #
################################################################################
class GAIL_FlowV2:
    def __init__(self, agent, state_dim, action_dim, device, lr=1e-3, beta_anti=1e-3, gamma_stab=0):
        self.device, self.agent = device, agent
        self.beta_anti, self.gamma_stab = beta_anti, gamma_stab
        self.field = CoupledFlowMatching(state_dim, action_dim).to(device)
        self.opt_field = torch.optim.Adam(self.field.parameters(), lr=lr)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def _update_fields(self, s_E, a_E, s_A, a_A):
        """One gradient step on FM + anti-div + Jacobian‑stab losses."""
        # ---------------- FM losses ----------------
        loss_E = self.field.c_fm_loss(s_E, a_E, role="expert")
        loss_A = self.field.c_fm_loss(s_A, a_A, role="agent")

        mix_s = torch.cat([s_E, s_A], 0)
        mix_a_raw = torch.cat([a_E, a_A], 0)
        t_mix = torch.rand_like(mix_a_raw[:, :1])

        mix_a = mix_a_raw.detach().requires_grad_(True)
        v_c, r_phi = self.field.net(mix_a, mix_s, t_mix)
        div_r = _hutchinson_div(mix_a, r_phi, k=4)
        loss_anti = div_r.square().mean()
        j_pen = _jacobian_frobenius(mix_a, v_c + r_phi).mean()
        loss = loss_E + loss_A + self.beta_anti * loss_anti + self.gamma_stab * j_pen
        self.opt_field.zero_grad()
        loss.backward()
        self.opt_field.step()

    # ------------------------------------------------------------------
    def _calc_reward(self, s, a):
        logp_E = self.field.log_prob(s, a, role="expert")  # [B]
        logp_A = self.field.log_prob(s, a, role="agent")  # [B]
        r = logp_E - logp_A
        r = (r - r.mean()) / (r.std() + 1e-8)
        return r.detach().cpu().numpy()

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s):
        agent_s_np = dynamic_convert(agent_s, self.state_dim)
        agent_a_np = dynamic_convert(agent_a, self.action_dim)

        s_A = torch.from_numpy(agent_s_np).to(self.device)
        a_A = torch.from_numpy(agent_a_np).to(self.device)
        if s_A.dim() == 1:
            s_A = s_A.unsqueeze(-1)
        if a_A.dim() == 1:
            a_A = a_A.unsqueeze(-1)

        expert_s_np = dynamic_convert(expert_s, self.state_dim)
        expert_a_np = dynamic_convert(expert_a, self.action_dim)
        s_E = torch.from_numpy(expert_s_np).to(self.device)
        a_E = torch.from_numpy(expert_a_np).to(self.device)
        if s_E.dim() == 1:
            s_E = s_E.unsqueeze(-1)
        if a_E.dim() == 1:
            a_E = a_E.unsqueeze(-1)

        self._update_fields(s_E.detach(), a_E.detach(), s_A.detach(), a_A.detach())
        rewards = self._calc_reward(s_A, a_A)
        rewards = rewards.squeeze()

        self.agent.update(
            dict(
                states=agent_s,
                actions=agent_a,
                rewards=rewards,
                next_states=next_s,
                dones=[False] * len(agent_s)
            )
        )


if __name__ == "__main__":
    x = torch.randn(3, 2, requires_grad=True)
    f = 2 * x
    print(_hutchinson_div(x, f, k=4))  # [4., 4., 4.]
    print(_jacobian_frobenius(x, f))  # [8., 8., 8.]

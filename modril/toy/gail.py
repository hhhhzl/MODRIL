import torch
import torch.nn as nn
from modril.toy.discriminators import Discriminator, MI_Estimator, FFJORDDensity, FlowMatching, CoupledFlowMatching, _jacobian_frobenius, _hutchinson_div
import numpy as np
from modril.modril.model_base_diffusion import MBDScore
from modril.toy.utils import dynamic_convert


class GAIL:
    def __init__(
            self,
            agent,
            state_dim,
            action_dim,
            hidden_dim,
            lr_d,
            epochs=5,
            device='cuda'
    ):
        self.discriminator = Discriminator(mode="gail", state_dim=state_dim, action_dim=action_dim,
                                           hidden_dim=hidden_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
        self.agent = agent
        self.device = device
        self.bce = nn.BCELoss()
        self.epochs = epochs
        self.state_dim = state_dim
        self.action_dim = action_dim

    def learn(
            self,
            expert_s,
            expert_a,
            agent_s,
            agent_a,
            next_s
    ):
        expert_s_arr = np.array(expert_s, dtype=np.float32)
        expert_a_arr = np.array(expert_a, dtype=np.float32)
        expert_states = torch.from_numpy(expert_s_arr).to(self.device)  # (batch, state_dim)
        expert_actions = torch.from_numpy(expert_a_arr).to(self.device)

        # agent_s maybe list of scalars（for static environment），also list of 1D vector（dynamic environment）
        # so reshape to (state_dim,)：
        agent_s_arr = dynamic_convert(agent_s, self.state_dim)
        agent_a_arr = dynamic_convert(agent_a, self.action_dim)  # (batch, action_dim)
        agent_states = torch.from_numpy(agent_s_arr).to(self.device)  # (batch, state_dim)
        agent_actions = torch.from_numpy(agent_a_arr).to(self.device)  # (batch, action_dim)

        # ------ Discriminator Training ------
        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        discriminator_loss = (
                self.bce(expert_prob, torch.ones_like(expert_prob))
                + self.bce(agent_prob, torch.zeros_like(agent_prob))
        )
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        with torch.no_grad():
            rewards = -torch.log(1 - agent_prob + 1e-8).cpu().numpy()
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards = rewards.squeeze()

        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': [False] * len(agent_s)
        }
        self.agent.update(transition_dict)


class DRAIL:
    def __init__(self, agent, state_dim, action_dim, device, disc_lr=1e-3, label_dim=10, epochs=5):
        self.discriminator = Discriminator(mode="mfd", state_dim=state_dim, action_dim=action_dim,
                                           label_dim=label_dim).to(device)
        self.opt = torch.optim.Adam(self.discriminator.parameters(), lr=disc_lr)
        self.agent = agent
        self.device = device
        self.bce = nn.BCELoss()
        self.epochs = epochs
        self.state_dim = state_dim
        self.action_dim = action_dim

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s):
        expert_s_arr, expert_a_arr = np.asarray(expert_s), np.asarray(expert_a)
        if expert_s_arr.ndim == 1:
            expert_s_arr = expert_s_arr.reshape(-1, 1)
        if expert_a_arr.ndim == 1:
            expert_a_arr = expert_a_arr.reshape(-1, 1)

        agent_s_arr, agent_a_arr = dynamic_convert(agent_s, self.state_dim), dynamic_convert(agent_a, self.action_dim)
        if agent_s_arr.ndim == 1:
            agent_s_arr = agent_s_arr.reshape(-1, 1)
        if agent_a_arr.ndim == 1:
            agent_a_arr = agent_a_arr.reshape(-1, 1)

        xs_E_arr = np.concatenate([expert_s_arr, expert_a_arr], axis=1)
        xs_A_arr = np.concatenate([agent_s_arr, agent_a_arr], axis=1)
        xs_E = torch.tensor(xs_E_arr, dtype=torch.float32, device=self.device)
        xs_A = torch.tensor(xs_A_arr, dtype=torch.float32, device=self.device)

        for _ in range(self.epochs):
            D_E = self.discriminator(xs_E)
            D_A = self.discriminator(xs_A)
            discriminator_loss = self.bce(D_E, torch.ones_like(D_E)) + \
                                 self.bce(D_A, torch.zeros_like(D_A))
            self.opt.zero_grad()
            discriminator_loss.backward()
            self.opt.step()

        with torch.no_grad():
            rewards = self.discriminator.get_reward(xs_A).cpu().numpy()
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards = rewards.squeeze()

        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards.tolist() if isinstance(rewards, np.ndarray) else rewards,
            'next_states': next_s,
            'dones': [False] * len(agent_s)
        }
        self.agent.update(transition_dict)


class GAIL_MI:
    def __init__(
            self,
            agent,
            state_dim,
            action_dim,
            device,
            mode="mine",
            disc_lr=1e-4
    ):
        self.mi_est = MI_Estimator(
            state_dim,
            action_dim,
            device,
            lr=disc_lr,
            mode=mode
        )
        self.agent = agent
        self.device = device

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s):
        rewards = self.mi_est.estimate_and_update(expert_s, expert_a, agent_s, agent_a)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        rewards = rewards.squeeze()

        transition_dict = dict(states=agent_s,
                               actions=agent_a,
                               rewards=rewards,
                               next_states=next_s,
                               dones=[False] * len(agent_s))
        self.agent.update(transition_dict)


class GAIL_Flow:
    """
    mode = 'ffjord'  or  'fm'
    density_E  -- offline  pre-trained
    density_A  -- online   1-2 step
    """

    def __init__(self, agent, state_dim, action_dim, device, mode="ffjord", lr=1e-3):
        self.mode = mode
        dim = state_dim + action_dim
        if mode == "ffjord":
            self.E = FFJORDDensity(dim, device).to(device)
            self.A = FFJORDDensity(dim, device).to(device)
            self.optA = torch.optim.Adam(self.A.parameters(), lr=lr)
        elif mode == "fm":
            self.E = FlowMatching(state_dim, action_dim, device).to(device)
            self.A = FlowMatching(state_dim, action_dim, device).to(device)
            self.optA = torch.optim.Adam(self.A.parameters(), lr=lr)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.agent = agent
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

    def _update_agent_density(self, s_A, a_A):
        """
        s_A: [B, state_dim]
        a_A: [B, action_dim]
        """
        if self.mode == "ffjord":
            xs_A = torch.cat([s_A, a_A], dim=1)  # [B, state_dim+action_dim]
            loss = self.A.nll(xs_A)
        else:
            loss = self.A.c_fm_loss(s_A, a_A)

        self.optA.zero_grad()
        loss.backward()
        self.optA.step()

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s):
        agent_s_np = dynamic_convert(agent_s, self.state_dim)
        agent_a_np = dynamic_convert(agent_a, self.action_dim)
        s_A = torch.from_numpy(agent_s_np).to(self.device)  # shape: [B] or [B, state_dim]
        a_A = torch.from_numpy(agent_a_np).to(self.device)

        if s_A.dim() == 1:
            s_A = s_A.unsqueeze(-1)
        if a_A.dim() == 1:
            a_A = a_A.unsqueeze(-1)

        # xs_E = torch.tensor(np.concatenate([expert_s, expert_a], 1), dtype=torch.float32, device=self.device)
        xs_A = torch.cat([s_A, a_A], dim=1)
        self._update_agent_density(s_A.detach(), a_A.detach())

        # with torch.no_grad():
        logp_E = self.E.log_prob(xs_A).detach().cpu().numpy()
        logp_A = self.A.log_prob(xs_A).detach().cpu().numpy()
        rewards = logp_E - logp_A
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
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


class EnergyGAIL:
    """
    r(s,a) = -E(s,a)，
    """

    def __init__(self, agent, state_dim, action_dim, hidden_dim, device='cuda'):
        self.agent = agent
        self.device = device

        self.E = None
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s):
        agent_s_np = dynamic_convert(agent_s, self.state_dim)
        agent_a_np = dynamic_convert(agent_a, self.action_dim)
        if agent_s_np.ndim == 1:
            agent_s_np = agent_s_np.reshape(-1, 1)
        if agent_a_np.ndim == 1:
            agent_a_np = agent_a_np.reshape(-1, 1)

        s_A = torch.from_numpy(agent_s_np).to(self.device)  # (B, state_dim)
        a_A = torch.from_numpy(agent_a_np).to(self.device)  # (B, action_dim)
        x_A = torch.cat([s_A, a_A], dim=1)  # (B, state_dim + action_dim)

        with torch.no_grad():
            energy = self.E.forward(x_A).cpu().numpy()  # (B,)

        rewards = -energy
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        rewards = rewards.squeeze().tolist()

        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': [False] * len(agent_s)
        }
        self.agent.update(transition_dict)


# ============================================================
#  Model-Based Diffusion - occupancy-reward
# ============================================================
class GAIL_MBD:
    """"""

    def __init__(
            self,
            agent,
            env,
            state_dim,
            action_dim,
            steps,
            env_name: str,
            device='cuda',
            mbd_kwargs: dict | None = None
    ):
        self.mbd = MBDScore(env, env_name, steps=steps, device=device, state_dim=state_dim, action_dim=action_dim, **(mbd_kwargs or {}))
        self.agent = agent
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.steps = steps

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s):
        expert_s_arr = np.asarray(expert_s, dtype=np.float32)
        expert_a_arr = np.asarray(expert_a, dtype=np.float32)
        if expert_s_arr.ndim == 1:
            expert_s_arr = expert_s_arr.reshape(-1, 1)
        if expert_a_arr.ndim == 1:
            expert_a_arr = expert_a_arr.reshape(-1, 1)

        # random window to align with agent
        start = np.random.randint(0, len(expert_s) - self.steps + 1)
        idx = slice(start, start + self.steps)

        xs_E_arr = np.concatenate([expert_s_arr[idx], expert_a_arr[idx]], axis=1)  # shape = (N, 2)
        xs_E = torch.tensor(xs_E_arr, dtype=torch.float32, device=self.device)

        agent_s_arr = dynamic_convert(agent_s, self.state_dim)
        agent_a_arr = np.asarray(agent_a, dtype=np.float32)
        if agent_s_arr.ndim == 1:
            agent_s_arr = agent_s_arr.reshape(-1, 1)
        if agent_a_arr.ndim == 1:
            agent_a_arr = agent_a_arr.reshape(-1, 1)
        xs_A_arr = np.concatenate([agent_s_arr, agent_a_arr], axis=1)  # shape = (M, 2)
        xs_A = torch.tensor(xs_A_arr, dtype=torch.float32, device=self.device)

        rewards = self.mbd.compute_reward(xs_E, xs_A)

        scale = 4.0
        clip = 10.0
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        rewards = np.clip(rewards * scale, -clip, clip)

        self.agent.update(dict(
            states=agent_s,
            actions=agent_a,
            rewards=rewards,
            next_states=next_s,
            dones=[False] * len(agent_s)
        ))

class GAIL_FlowShare:
    def __init__(self, agent, state_dim, action_dim, device, lr=1e-3, beta_anti=1e-6, gamma_stab=1e-8):
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
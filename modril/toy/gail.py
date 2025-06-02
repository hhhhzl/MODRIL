import torch
import torch.nn.functional as F
import torch.nn as nn
from modril.toy.discriminators import Discriminator, MI_Estimator, FFJORDDensity, FlowMatching
import numpy as np
from modril.modril.model_base_diffusion import MBDScore


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

    def learn(
            self,
            expert_s,
            expert_a,
            agent_s,
            agent_a,
            next_s
    ):
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(self.device)
        expert_actions = torch.tensor(expert_a, dtype=torch.float).to(self.device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(self.device)
        agent_actions = torch.tensor(agent_a, dtype=torch.float).to(self.device)

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        discriminator_loss = self.bce(expert_prob, torch.ones_like(expert_prob)) + self.bce(agent_prob,
                                                                                            torch.zeros_like(
                                                                                                agent_prob))
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

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s):
        expert_s_arr, expert_a_arr = np.asarray(expert_s), np.asarray(expert_a)
        if expert_s_arr.ndim == 1:
            expert_s_arr = expert_s_arr.reshape(-1, 1)
        if expert_a_arr.ndim == 1:
            expert_a_arr = expert_a_arr.reshape(-1, 1)

        agent_s_arr, agent_a_arr = np.asarray(agent_s), np.asarray(agent_a)
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
            self.E = FFJORDDensity(dim).to(device)
            self.A = FFJORDDensity(dim).to(device)
            self.optA = torch.optim.Adam(self.A.parameters(), lr=lr)
        elif mode == "fm":
            self.E = FlowMatching(state_dim, action_dim, device).to(device)
            self.A = FlowMatching(state_dim, action_dim, device).to(device)
            self.optA = torch.optim.Adam(self.A.parameters(), lr=lr)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.agent = agent
        self.device = device

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
        s_A = torch.tensor(agent_s, dtype=torch.float32, device=self.device)
        a_A = torch.tensor(agent_a, dtype=torch.float32, device=self.device)

        if s_A.dim() == 1:
            s_A = s_A.unsqueeze(-1)
        if a_A.dim() == 1:
            a_A = a_A.unsqueeze(-1)

        # xs_E = torch.tensor(np.concatenate([expert_s, expert_a], 1), dtype=torch.float32, device=self.device)
        xs_A = torch.cat([s_A, a_A], dim=1)
        self._update_agent_density(s_A.detach(), a_A.detach())

        # with torch.no_grad():
        logp_E = self.E.log_prob(xs_A).detach().numpy()
        logp_A = self.A.log_prob(xs_A).detach().numpy()
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


# ============================================================
#  Model-Based Diffusion - occupancy-reward
# ============================================================
class GAIL_MBD:
    """"""

    def __init__(
            self,
            agent,
            env,
            steps,
            env_name: str,
            device='cuda',
            mbd_kwargs: dict | None = None
    ):
        self.mbd = MBDScore(env, env_name, steps=steps, device=device, **(mbd_kwargs or {}))
        self.agent = agent
        self.device = device

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s):
        xs_E = torch.tensor(np.stack([expert_s, expert_a], 1), dtype=torch.float32).view(-1, 2).to(self.device)
        xs_A = torch.tensor(np.stack([agent_s, agent_a], 1), dtype=torch.float32).view(-1, 2).to(self.device)
        rewards = self.mbd.compute_reward(xs_E, xs_A)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        self.agent.update(dict(
            states=agent_s,
            actions=agent_a,
            rewards=rewards,
            next_states=next_s,
            dones=[False] * len(agent_s)
        ))

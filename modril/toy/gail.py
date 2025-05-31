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
        expert_states = torch.tensor(expert_s, dtype=torch.float).view(-1, 1).to(self.device)
        expert_actions = torch.tensor(expert_a, dtype=torch.float).view(-1, 1).to(self.device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).view(-1, 1).to(self.device)
        agent_actions = torch.tensor(agent_a, dtype=torch.float).view(-1, 1).to(self.device)

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        # print("D_expert", expert_prob.mean(), "D_agent", agent_prob.mean())
        discriminator_loss = self.bce(expert_prob, torch.ones_like(expert_prob)) + self.bce(agent_prob,
                                                                                            torch.zeros_like(
                                                                                                agent_prob))
        # print("d_loss", discriminator_loss)
        # discriminator_loss = F.binary_cross_entropy(agent_prob, torch.ones_like(agent_prob)) + F.binary_cross_entropy(expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        # wandb.log({'discriminator_loss':discriminator_loss})

        with torch.no_grad():
            rewards = -torch.log(1 - agent_prob + 1e-8).cpu().numpy()
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        # wandb.log({'agent_prob':agent_prob[0]})
        # wandb.log({'reward for policy training':rewards[0]})
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
        xs_E = torch.tensor(np.stack([expert_s.squeeze(), expert_a.squeeze()], axis=1), dtype=torch.float32,
                            device=self.device)
        xs_A = torch.tensor(np.stack([agent_s, agent_a], axis=1), dtype=torch.float32, device=self.device)
        # now (batch, 2)
        xs_E = xs_E.view(xs_E.size(0), -1)
        xs_A = xs_A.view(xs_A.size(0), -1)

        for _ in range(self.epochs):
            D_E = self.discriminator(xs_E)
            D_A = self.discriminator(xs_A)
            discriminator_loss = self.bce(D_E, torch.ones_like(D_E)) + self.bce(D_A, torch.zeros_like(D_A))
            # print("D_expert", D_E.mean(), "D_agent", D_A.mean())
            self.opt.zero_grad()
            discriminator_loss.backward()
            self.opt.step()

        with torch.no_grad():
            rewards = self.discriminator.get_reward(xs_A).cpu().numpy()
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
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
        transition_dict = dict(states=agent_s,
                               actions=agent_a,
                               rewards=rewards,
                               next_states=next_s,
                               dones=[False] * len(agent_s))
        self.agent.update(transition_dict)


class GAIL_Flow:
    """
    mode = 'ffjord'  or  'fm'
    density_E  -- offline  pre-trained   (固定)
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
        # xs_E = torch.tensor(np.stack([expert_s, expert_a], 1), dtype=torch.float32, device=self.device)
        s_A = torch.tensor(agent_s, dtype=torch.float32, device=self.device)
        a_A = torch.tensor(agent_a, dtype=torch.float32, device=self.device)
        xs_A = torch.tensor(np.stack([agent_s, agent_a], 1), dtype=torch.float32, device=self.device)
        if s_A.dim() == 1:
            s_A = s_A.unsqueeze(-1)
        if a_A.dim() == 1:
            a_A = a_A.unsqueeze(-1)
        self._update_agent_density(s_A.detach(), a_A.detach())

        with torch.no_grad():
            logp_E = self.E.log_prob(xs_A).cpu().numpy()
            logp_A = self.A.log_prob(xs_A).cpu().numpy()
            rewards = logp_E - logp_A
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

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
        rewards = self.mbd.compute_reward(expert_a, agent_a)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        self.agent.update(dict(
            states=agent_s,
            actions=agent_a,
            rewards=rewards,
            next_states=next_s,
            dones=[False] * len(agent_s)
        ))

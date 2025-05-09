import os
from rlf.algos.il.base_irl import BaseIRLAlgo
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler
import rlf.rl.utils as rutils
from collections import defaultdict
from rlf.baselines.common.running_mean_std import RunningMeanStd
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
from rlf.args import str2bool
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from rlf.exp_mgr.viz_utils import append_text_to_image
import math
from rlf.rl.utils import get_obs_shape, get_ac_dim
from modril.modril.model_base_diffusion import MBDScore


class MODRIL(NestedAlgo):
    def __init__(self, agent_updater=PPO()):
        super().__init__([MBDRewardModule(policy=agent_updater), agent_updater], 1)


class MBDRewardModule(BaseIRLAlgo):
    """ """

    def __init__(self, policy=None):
        super().__init__()
        self.policy = policy

    def init(self, policy, args):
        super().init(policy, args)
        state_dim = get_obs_shape(policy.obs_space)[0]
        action_dim = get_ac_dim(policy.action_space)
        self.mbd = MBDScore()

    def _compute_reward(self, agent_traj):
        expert_traj = self.expert_buffer.sample()
        reward = self.mbd.compute_reward(expert_traj, agent_traj)
        return reward

    def update(self, storage):
        agent_traj = storage.sample_trajectory()
        expert_traj = self.expert_buffer.sample()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"score_loss": loss.item()}

    def plot_reward_map(self, i):
        x = torch.linspace(0, 10, 100)
        y = torch.linspace(-2, 2, 100)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        X = X.reshape(-1, 1).to(self.args.device)
        Y = Y.reshape(-1, 1).to(self.args.device)
        with torch.no_grad():
            s = self._compute_disc_val(X, Y)
            eps = 1e-20
            if self.args.reward_type == 'airl':
                reward = (s + eps).log() - (1 - s + eps).log()
            elif self.args.reward_type == 'gail':
                reward = (s + eps).log()
            elif self.args.reward_type == 'raw':
                reward = s
            elif self.args.reward_type == 'airl-positive':
                reward = (s + eps).log() - (1 - s + eps).log() + 20
            elif self.args.reward_type == 'revise':
                d_x = (s + eps).log()
                reward = d_x + (-1 - (-d_x).log())
            else:
                raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
            reward = reward.view(100, 100).cpu().numpy().T

        plt.figure(figsize=(8, 5))
        plt.imshow(reward, extent=[0, 10, -2, 2], cmap="jet", origin="lower", aspect="auto")
        plt.colorbar()
        file_path = "./data/imgs/" + self.args.prefix + "_reward_map.png"
        plt.savefig(file_path)
        return file_path

    def plot_disc_val_map(self, i):
        x = torch.linspace(0, 10, 100)
        y = torch.linspace(-2, 2, 100)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        X = X.reshape(-1, 1).to(self.args.device)
        Y = Y.reshape(-1, 1).to(self.args.device)
        with torch.no_grad():
            rewards = []
            for _ in range(10):
                reward = self._compute_disc_val(X, Y).view(100, 100).cpu().numpy().T
                rewards.append(reward)
            reward = torch.tensor(rewards).mean(dim=0)
        plt.figure(figsize=(8, 5))
        plt.imshow(reward, extent=[0, 10, -2, 2], cmap="jet", origin="lower", aspect="auto")
        plt.colorbar()
        file_path = "./data/imgs/" + self.args.prefix + "_disc_val_map.png"
        plt.savefig(file_path)
        return file_path

    def _compute_expert_loss(self, expert_d, expert_batch):
        return F.binary_cross_entropy(expert_d,
                                      torch.ones(expert_d.shape).to(self.args.device))

    def _compute_agent_loss(self, agent_d, agent_batch):
        return F.binary_cross_entropy(agent_d,
                                      torch.zeros(agent_d.shape).to(self.args.device))

    def _update_reward_func(self, storage, gradient_clip=False, t=1):
        self.discrim_net.train()

        log_vals = defaultdict(lambda: 0)
        obsfilt = self.get_env_ob_filt()

        expert_sampler, agent_sampler = self._get_sampler(storage)
        if agent_sampler is None:
            # algo requested not to update this step
            return {}

        n = 0
        for epoch_i in range(self.args.n_drail_epochs):
            for expert_batch, agent_batch in zip(expert_sampler, agent_sampler):
                expert_batch, agent_batch = self._trans_batches(expert_batch, agent_batch)
                n += 1
                expert_d, agent_d, grad_pen = self._compute_discrim_loss(agent_batch, expert_batch, obsfilt)
                expert_loss = self._compute_expert_loss(expert_d, expert_batch)
                agent_loss = self._compute_agent_loss(agent_d, agent_batch)

                discrim_loss = expert_loss + agent_loss
                if self.args.disc_grad_pen != 0.0:
                    if t <= self.args.disc_grad_pen_period:
                        log_vals['grad_pen'] += grad_pen.item()
                        total_loss = discrim_loss + self.args.disc_grad_pen * grad_pen
                    else:
                        log_vals['grad_pen'] += 0
                        total_loss = discrim_loss
                else:
                    total_loss = discrim_loss

                self.opt.zero_grad()
                total_loss.backward()
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.discrim_net.parameters(), max_norm=1.0)
                self.opt.step()

                log_vals['discrim_loss'] += discrim_loss.item()
                log_vals['expert_loss'] += expert_loss.item()
                log_vals['agent_loss'] += agent_loss.item()
                log_vals['expert_disc_val'] += expert_d.mean().item()
                log_vals['agent_disc_val'] += agent_d.mean().item()
                log_vals['agent_reward'] += ((agent_d + 1e-20).log() - (1 - agent_d + 1e-20).log()).mean().item()
                log_vals['dm_update_data'] += len(expert_batch)
                self.step += self.expert_train_loader.batch_size
        for k in log_vals:
            if k[0] != '_':
                log_vals[k] /= n
        if self.args.env_name[:4] == "Sine" and (self.step // (self.expert_train_loader.batch_size * n)) % 100 == 1:
            # log_vals["_reward_map"] = self.plot_reward_map(self.step)
            log_vals["_disc_val_map"] = self.plot_disc_val_map(self.step)
        log_vals['dm_update_data'] *= n
        return log_vals

    def _compute_discrim_reward(self, storage, step, add_info):
        state = self._trans_agent_state(storage.get_obs(step))
        action = storage.actions[step]
        action = rutils.get_ac_repr(self.action_space, action)
        s = self._compute_disc_val(state, action)
        eps = 1e-20
        if self.args.reward_type == 'airl':
            reward = (s + eps).log() - (1 - s + eps).log()
        elif self.args.reward_type == 'gail':
            reward = (s + eps).log()
        elif self.args.reward_type == 'raw':
            reward = s
        elif self.args.reward_type == 'airl-positive':
            reward = (s + eps).log() - (1 - s + eps).log() + 20
        elif self.args.reward_type == 'revise':
            d_x = (s + eps).log()
            reward = d_x + (-1 - (-d_x).log())
        else:
            raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
        return reward

    def _get_reward(self, step, storage, add_info):
        masks = storage.masks[step]
        with torch.no_grad():
            self.discrim_net.eval()
            reward = self._compute_discrim_reward(storage, step, add_info)

            if self.args.drail_reward_norm:
                if self.returns is None:
                    self.returns = reward.clone()

                self.returns = self.returns * masks * self.args.gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8), {}
            else:
                return reward, {}

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides
        #########################################
        # New args
        parser.add_argument('--action-input', type=str2bool, default=False)
        parser.add_argument('--drail-reward-norm', type=str2bool, default=False)
        parser.add_argument('--drail-state-norm', type=str2bool, default=True)
        parser.add_argument('--drail-action-norm', type=str2bool, default=False)
        parser.add_argument('--disc-lr', type=float, default=0.0001)
        parser.add_argument('--disc-grad-pen', type=float, default=0.0)
        parser.add_argument('--disc-grad-pen-period', type=float, default=1.0)
        parser.add_argument('--expert-loss-rate', type=float, default=1.0)
        parser.add_argument('--agent-loss-rate', type=float, default=-1.0)
        parser.add_argument('--agent-loss-rate-scheduler', type=str2bool, default=False)
        parser.add_argument('--agent-loss-end', type=float, default=-1.1)
        parser.add_argument('--discrim-depth', type=int, default=4)
        parser.add_argument('--discrim-num-unit', type=int, default=128)
        parser.add_argument('--sample-strategy', type=str, default="random")
        parser.add_argument('--sample-strategy-value', type=int, default=250)
        parser.add_argument('--n-drail-epochs', type=int, default=1)
        parser.add_argument('--label-dim', type=int, default=10)
        parser.add_argument('--test-sine-env', type=str2bool, default=False)
        parser.add_argument('--deeper-ddpm', type=str2bool, default=False)
        parser.add_argument('--reward-type', type=str, default='airl', help="""
                One of [Drail]. Changes the reward computation. Does
                not change training.
                """)

    def load_resume(self, checkpointer):
        super().load_resume(checkpointer)
        self.opt.load_state_dict(checkpointer.get_key('drail_disc_opt'))
        self.discrim_net.load_state_dict(checkpointer.get_key('drail_disc'))

    def save(self, checkpointer):
        super().save(checkpointer)
        checkpointer.save_key('drail_disc_opt', self.opt.state_dict())
        checkpointer.save_key('drail_disc', self.discrim_net.state_dict())


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

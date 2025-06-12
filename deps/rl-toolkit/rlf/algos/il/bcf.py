import copy
import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

import rlf.algos.utils as autils
import rlf.rl.utils as rutils
from rlf.algos.il.base_il import BaseILAlgo
from rlf.storage.base_storage import BaseStorage
from rlf.args import str2bool


class BehaviorCloneFlowMatching(BaseILAlgo):
    """
    BehaviorCloneFlowMatching (BCFM):
    Trains a conditional velocity field v_theta(t, s, x_t)
    to match expert actions via flow-matching loss in action space.
    """

    def __init__(self, set_arg_defs=True):
        super().__init__()
        self.set_arg_defs = set_arg_defs

    def init(self, policy, args):
        super().init(policy, args)
        # action dimension
        self.num_epochs = 0
        self.action_dim = rutils.get_ac_dim(self.policy.action_space)
        # tuning noise / normalization
        if self.args.bc_state_norm:
            self.norm_mean = self.expert_stats["state"][0]
            self.norm_var = torch.pow(self.expert_stats["state"][1], 2)
        else:
            self.norm_mean = None
            self.norm_var = None
        self.num_bc_updates = 0

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if args.bc_state_norm:
            settings.state_fn = self._norm_state
        return settings

    def _norm_state(self, x):
        obs = torch.clamp(
            (rutils.get_def_obs(x) - self.norm_mean)
            / torch.pow(self.norm_var + 1e-8, 0.5),
            -10.0, 10.0
        )
        if isinstance(x, dict):
            x['observation'] = obs
            return x
        return obs

    def get_num_updates(self):
        if self.exp_generator is None:
            return len(self.expert_train_loader) * self.args.bc_num_epochs
        else:
            return self.args.exp_gen_num_trans * self.args.bc_num_epochs

    def get_completed_update_steps(self, num_updates):
        return num_updates * self.args.traj_batch_size

    def _reset_data_fetcher(self):
        super()._reset_data_fetcher()
        self.num_epochs += 1

    def full_train(self, update_iter=0):
        action_loss = []
        prev_num = 0

        with tqdm(total=self.args.bc_num_epochs) as pbar:
            while self.num_epochs < self.args.bc_num_epochs:
                super().pre_update(self.num_bc_updates)
                log_vals = self._bc_step(False)
                action_loss.append(log_vals['_pr_flow_loss'])

                pbar.update(self.num_epochs - prev_num)
                prev_num = self.num_epochs

        # optional plotting
        rutils.plot_line(
            action_loss,
            f"flow_loss_{update_iter}.png",
            self.args.vid_dir,
            not self.args.no_wb,
            self.get_completed_update_steps(self.update_i)
        )
        self.num_epochs = 0

    def pre_update(self, cur_update):
        # override no-op to prevent BC decay
        pass

    def _bc_step(self, decay_lr):
        if decay_lr:
            super().pre_update(self.num_bc_updates)

        expert_batch = self._get_next_data()
        if expert_batch is None:
            self._reset_data_fetcher()
            expert_batch = self._get_next_data()
        states, actions = self._get_data(expert_batch)



        # flow matching
        x0 = torch.randn_like(actions)
        batch_size = actions.shape[0]
        t = torch.rand(batch_size, 1, device=actions.device)
        x_t = (1 - t) * x0 + t * actions
        v_pred = self.policy(x_t, states, t)
        v_target = actions - x0
        flow_loss = F.mse_loss(v_pred, v_target)

        # t=1 bc loss
        t1 = torch.ones_like(t)
        x_t1 = actions
        v_end = self.policy(x_t1, states, t1)
        a_pred = x0 + v_end
        action_loss = F.mse_loss(a_pred, actions)

        coef = getattr(self.args, "bc_coef", 1.0)
        loss = flow_loss + coef * action_loss

        self._standard_step(loss)
        self.num_bc_updates += 1
        log_dict = {
            "_pr_flow_loss": flow_loss.item(),
            "action_loss": action_loss.item(),
        }
        return log_dict

    def _get_data(self, batch):
        states = batch["state"].to(self.args.device)
        if self.args.bc_state_norm:
            states = self._norm_state(states)

        if self.args.bc_noise is not None:
            add_noise = torch.randn(states.shape) * self.args.bc_noise
            states += add_noise.to(self.args.device)
            states = states.detach()

        true_actions = batch["actions"].to(self.args.device)
        true_actions = self._adjust_action(true_actions)
        return states, true_actions

    def update(self, storage, args, beginning, t):
        top_log_vals = super().update(storage)
        log_vals = self._bc_step(True)
        log_vals.update(top_log_vals)
        return log_vals

    def get_storage_buffer(self, policy, envs, args):
        # no environment rollout
        return BaseStorage()

    def get_add_args(self, parser):
        if not self.set_arg_defs:
            self.set_arg_prefix('bcf')

        super().get_add_args(parser)

        if self.set_arg_defs:
            parser.add_argument('--num-processes', type=int, default=1)
            parser.add_argument('--num-steps', type=int, default=0)
            ADJUSTED_INTERVAL = 200
            parser.add_argument("--log-interval", type=int, default=ADJUSTED_INTERVAL)
            parser.add_argument(
                "--save-interval", type=int, default=100 * ADJUSTED_INTERVAL
            )
            parser.add_argument(
                "--eval-interval", type=int, default=100 * ADJUSTED_INTERVAL
            )
        parser.add_argument("--no-wb", default=False, action="store_true")

        #########################################
        # New args
        parser.add_argument("--bc-num-epochs", type=int, default=1)
        parser.add_argument("--bc-coef", type=int, default=1)
        parser.add_argument("--bc-state-norm", type=str2bool, default=False)
        parser.add_argument("--bc-noise", type=float, default=None)

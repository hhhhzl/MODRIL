from rlf.algos.il.base_irl import BaseIRLAlgo
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rlf.rl.utils as rutils
from collections import defaultdict
from rlf.baselines.common.running_mean_std import RunningMeanStd
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
import torch.optim as optim
import numpy as np
from rlf.exp_mgr.viz_utils import append_text_to_image
from deps.baselines.ebil.deen import DEENDensity
import os


def str2bool(v):
    return v.lower() == "true"

class EBIL(NestedAlgo):
    def __init__(self, agent_updater=PPO()):
        super().__init__([EnergyDensity(), agent_updater], 1)


class EnergyDensity(BaseIRLAlgo):
    def __init__(self):
        super().__init__()
        self.step = 0

    def _create_energy_net(self):
        # Determine state and action dimensions from policy
        ob_shape = rutils.get_obs_shape(self.policy.obs_space)
        action_dim = rutils.get_ac_dim(self.action_space)
        state_dim = ob_shape[0]
        
        # Initialize energy network
        self.energy_net = DEENDensity(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.args.hidden_dim,
            sigma=self.args.sigma
        ).to(self.args.device)
        # Load pretrained weights if provided
        if self.args.energy_path:
            self.energy_net.load_state_dict(torch.load(self.args.energy_path, map_location=self.args.device))

    def init(self, policy, args):
        super().init(policy, args)
        self.action_space = self.policy.action_space

        self._create_energy_net()
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.opt = optim.Adam(self.energy_net.parameters(), lr=self.args.disc_lr)

    def _get_sampler(self, storage):
        agent_experience = storage.get_generator(None, mini_batch_size=self.expert_train_loader.batch_size)
        return self.expert_train_loader, agent_experience

    def _trans_batches(self, expert_batch, agent_batch):
        return expert_batch, agent_batch

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if not args.ebil_state_norm:
            settings.ret_raw_obs = True
        settings.mod_render_frames_fn = self.mod_render_frames
        return settings

    def mod_render_frames(self, frame, env_cur_obs, env_cur_action, env_cur_reward,
                          env_next_obs, **kwargs):
        use_cur_obs = rutils.get_def_obs(env_cur_obs)
        use_cur_obs = torch.FloatTensor(use_cur_obs).unsqueeze(0).to(self.args.device)

        if env_cur_action is not None:
            use_action = torch.FloatTensor(env_cur_action).unsqueeze(0).to(self.args.device)
            disc_val = self._compute_disc_val(use_cur_obs, use_action).item()
        else:
            disc_val = 0.0

        frame = append_text_to_image(frame, [
            "Discrim: %.3f" % disc_val,
            "Reward: %.3f" % (env_cur_reward if env_cur_reward is not None else 0.0)
        ])
        return frame

    def _norm_expert_state(self, state, obsfilt):
        if not self.args.ebil_state_norm:
            return state
        state = state.cpu().numpy()

        if obsfilt is not None:
            state = obsfilt(state, update=False)
        state = torch.tensor(state).to(self.args.device)
        return state

    def _trans_agent_state(self, state, other_state=None):
        if not self.args.ebil_state_norm:
            if isinstance(state, dict):
                if other_state is None:
                    return state['raw_obs']
                return other_state['raw_obs']
            else:
                return state
        return rutils.get_def_obs(state)

    # rewards
    def _compute_energy_reward(self, storage, step, add_info):
        state = self._trans_agent_state(storage.get_obs(step)).to(self.args.device)
        action = storage.actions[step].to(self.args.device)
        action = rutils.get_ac_repr(self.action_space, action)

        inp = torch.cat([state, action], dim=-1)
        energy = self.energy_net(inp).unsqueeze(-1)
        eps = 1e-20
        s = torch.sigmoid(energy)

        if self.args.reward_type == 'norm':
            reward = (s + eps).log() - (1 - s + eps).log()
        elif self.args.reward_type == 'exp':
            exp_e = torch.exp(energy)
            reward = - exp_e / (1 + exp_e)
        elif self.args.reward_type == 'raw':
            reward = - energy * 1 + 0  # h(-e) in paper, where h is a monotonic increasing linear function
        elif self.args.reward_type == 'clip':
            reward = torch.clamp(-energy, min=-20.0, max=20.0)
        else:
            raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
        return reward

    def _get_reward(self, step, storage, add_info):
        masks = storage.masks[step]
        with torch.no_grad():
            self.energy_net.eval()
            reward = self._compute_energy_reward(storage, step, add_info)

            if self.args.ebil_reward_norm:
                if self.returns is None:
                    self.returns = reward.clone()

                self.returns = self.returns * masks * self.args.gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8), {}
            else:
                return reward, {}

    def _compute_energy_loss(self, agent_batch, expert_batch, obsfilt):
        expert_states = self._norm_expert_state(expert_batch['state'], obsfilt)
        expert_actions = expert_batch['action'].to(self.args.device)
        expert_actions = self._adjust_action(expert_actions)
        expert_actions = rutils.get_ac_repr(self.action_space, expert_actions)
        expert_inp = torch.cat([expert_states, expert_actions], dim=-1)

        agent_states = self._trans_agent_state(agent_batch['state'], agent_batch['other_state'] if 'other_state' in agent_batch else None)
        agent_actions = agent_batch['actions'].to(self.args.device)
        agent_actions = rutils.get_ac_repr(self.action_space, agent_actions)
        agent_inp = torch.cat([agent_states, agent_actions], dim=-1)
        
        expert_d = self.energy_net(expert_inp)
        agent_d = self.energy_net(agent_inp)
        expert_loss = self.args.expert_loss_rate * F.binary_cross_entropy_with_logits(
            expert_d, torch.ones_like(expert_d)
        )
        agent_loss = self.args.agent_loss_rate * F.binary_cross_entropy_with_logits(
            agent_d, torch.zeros_like(agent_d)
        )
        deen_loss = self.args.deen_loss_rate * self.energy_net.deen_loss(expert_inp)
        total_loss = expert_loss + agent_loss + deen_loss
        return total_loss, expert_loss, agent_loss, deen_loss

    def _compute_expert_loss(self, expert_d, expert_batch):
        return F.binary_cross_entropy(expert_d,
                                      torch.ones(expert_d.shape).to(self.args.device))

    def _compute_agent_loss(self, agent_d, agent_batch):
        return F.binary_cross_entropy(agent_d,
                                      torch.zeros(agent_d.shape).to(self.args.device))

    def _update_reward_func(self, storage, *args):
        self.energy_net.train()
        log_vals = defaultdict(float)
        obsfilt = self.get_env_ob_filt()
        # Get expert and agent samplers
        expert_sampler, agent_sampler = self._get_sampler(storage)
        if agent_sampler is None:
            return {}
        count = 0
        for _ in range(self.args.n_ebil_epochs):
            for expert_batch, agent_batch in zip(expert_sampler, agent_sampler):
                expert_batch, agent_batch = self._trans_batches(expert_batch, agent_batch)
                loss, e_loss, a_loss, d_loss = self._compute_energy_loss(expert_batch, agent_batch, obsfilt)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.energy_net.parameters(), self.args.disc_grad_pen)
                self.opt.step()
                # Accumulate logs
                log_vals['energy_loss'] += d_loss.item()
                log_vals['expert_loss'] += e_loss.item()
                log_vals['agent_loss'] += a_loss.item()
                self.step += 1
                count += 1

        for k in log_vals:
            log_vals[k] /= max(count, 1)
        if self.args.env_name.startswith("Sine") and (self.step % 100 == 0):
            # log_vals['_reward_map'] = self.plot_reward_map(self.step)
            log_vals['_disc_val_map'] = self.plot_disc_val_map(self.step)
            log_vals['_energy_map'] = self.plot_energy_field(self.step)

        return log_vals

    def _compute_disc_val(self, state, action):
        inp = torch.cat([state, action], dim=-1)
        return self.energy_net.log_energy(inp)

    def plot_disc_val_map(self, i):
        x = torch.linspace(0, 10, 100)
        y = torch.linspace(-2, 2, 100)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        X = X.reshape(-1, 1).to(self.args.device)
        Y = Y.reshape(-1, 1).to(self.args.device)
        with torch.no_grad():
            reward = self._compute_disc_val(X, Y).view(100, 100).cpu().numpy().T
        plt.figure(figsize=(8, 5))
        plt.imshow(reward, extent=[0, 10, -2, 2], cmap="jet", origin="lower", aspect="auto")
        plt.colorbar()
        file_path = "./data/imgs/" + self.args.prefix + "_disc_val_map.png"
        folder = os.path.dirname(file_path)
        os.makedirs(folder, exist_ok=True)
        plt.savefig(file_path)
        return file_path

    def plot_reward_map(self, i):
        x = torch.linspace(0, 10, 100)
        y = torch.linspace(-2, 2, 100)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        X = X.reshape(-1, 1).to(self.args.device)
        Y = Y.reshape(-1, 1).to(self.args.device)
        with torch.no_grad():
            d_val = self._compute_disc_val(X, Y)
            s = torch.sigmoid(d_val)
            eps = 1e-20
            if self.args.reward_type == 'norm':
                reward = (s + eps).log() - (1 - s + eps).log()
            elif self.args.reward_type == 'exp':
                exp_e = torch.exp(d_val)
                reward = - exp_e / (1 + exp_e)
            elif self.args.reward_type == 'raw':
                reward = - d_val * 1 + 0  # h(-e) in paper, where h is a monotonic increase linear function
            elif self.args.reward_type == 'clip':
                reward = torch.clamp(-d_val, min=-20.0, max=20.0)
            else:
                raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
            reward = reward.view(100, 100).cpu().numpy().T

        plt.figure(figsize=(8, 5))
        plt.imshow(reward, extent=[0, 10, -2, 2], cmap="jet", origin="lower", aspect="auto")
        plt.colorbar()
        file_path = "./data/imgs/" + self.args.prefix + "_reward_map.png"
        folder = os.path.dirname(file_path)
        os.makedirs(folder, exist_ok=True)
        plt.savefig(file_path)
        return file_path

    def plot_energy_field(self, i):
        # Generate grid over state-action space
        x = torch.linspace(0, 10, 100)
        y = torch.linspace(-2, 2, 100)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        X_flat = X.reshape(-1, 1).to(self.args.device)
        Y_flat = Y.reshape(-1, 1).to(self.args.device)
        # Compute energy values
        with torch.no_grad():
            E = self.energy_net.log_energy(torch.cat([X_flat, Y_flat], dim=1))
            field = E.view(100, 100).cpu().numpy().T
        # Plot
        plt.figure(figsize=(8, 5))
        plt.imshow(field, extent=[0, 10, -2, 2], cmap="jet", origin="lower", aspect="auto")
        plt.colorbar()
        file_path = "./data/imgs/" + self.args.prefix + "_energy_map.png"
        folder = os.path.dirname(file_path)
        os.makedirs(folder, exist_ok=True)
        plt.savefig(file_path)
        return file_path

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides
        #########################################
        # New args
        parser.add_argument('--n-ebil-epochs', type=int, default=1)
        parser.add_argument('--ebil-state-norm', type=str2bool, default=True)
        parser.add_argument('--ebil-reward-norm', type=str2bool, default=True)
        parser.add_argument('--energy-path', type=str, default=None)
        parser.add_argument('--energy-depth', type=int, default=4)
        parser.add_argument('--hidden-dim', type=int, default=256)
        parser.add_argument('--sigma', type=int, default=0.1)
        parser.add_argument('--disc-lr', type=float, default=1e-4)
        parser.add_argument('--disc-grad-pen', type=float, default=0.0)
        parser.add_argument('--expert-loss-rate', type=float, default=0)
        parser.add_argument('--agent-loss-rate', type=float, default=0)
        parser.add_argument('--deen-loss-rate', type=float, default=1.0)
        parser.add_argument('--reward-type', type=str, default='raw', help="""
                Changes the reward computation. Does
                not change training.
                """)

    def load_resume(self, checkpointer):
        super().load_resume(checkpointer)
        self.opt.load_state_dict(checkpointer.get_key('energy_opt'))
        self.energy_net.load_state_dict(checkpointer.get_key('energy'))

    def save(self, checkpointer):
        super().save(checkpointer)
        checkpointer.save_key('energy_opt', self.opt.state_dict())
        checkpointer.save_key('energy', self.energy_net.state_dict())

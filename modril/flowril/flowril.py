from rlf.algos.il.base_irl import BaseIRLAlgo
import torch
import matplotlib.pyplot as plt
import rlf.rl.utils as rutils
from collections import defaultdict
from rlf.baselines.common.running_mean_std import RunningMeanStd
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
import torch.optim as optim
import numpy as np
from rlf.exp_mgr.viz_utils import append_text_to_image
from modril.flowril.pretrain import FlowMatching, CoupledResidualFM, _jacobian_frobenius, CMAP, PLOT_KW, ALPHA_BG
from modril.flowril.networks import GradNorm2D
import os


def str2bool(v):
    return v.lower() == "true"


class FLOWRIL(NestedAlgo):
    def __init__(self, agent_updater=PPO()):
        super().__init__([FlowMatchingEstimation(), agent_updater], 1)


class FlowMatchingEstimation(BaseIRLAlgo):
    def __init__(self):
        super().__init__()
        self.step = 0
        self._vector_plot_path = []
        self._reward_plot_path = []
        self._disc_plot_path = []

    def _create_flow_net(self):
        # Determine state and action dimensions from policy
        ob_shape = rutils.get_obs_shape(self.policy.obs_space)
        self.state_dim = ob_shape[0]
        self.action_dim = rutils.get_ac_dim(self.action_space)

        # Initialize flow network
        if self.args.option == "scrf":
            self.flow_net = CoupledResidualFM(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.args.hidden_dim,
            ).to(self.args.device)
            # Load pretrained weights if provided
            if self.args.flow_path:
                self.flow_net.load_state_dict(torch.load(self.args.flow_path, map_location=self.args.device))
                for p in self.flow_net.parameters():
                    p.requires_grad = False

            self.opt = optim.Adam(self.flow_net.parameters(), lr=self.args.disc_lr)
            self.params = list(self.flow_net.parameters())
            if self.args.params_autotune:
                self.gradnorm = GradNorm2D(
                    beta_init=self.args.loss_anti,
                    gamma_init=self.args.loss_stable,
                    alpha=0.1,
                    warmup_steps=100,
                    update_freq=10,
                    ema_decay=0.9,
                    beta_min=1e-8,
                    beta_max=1e1,
                    gamma_min=1e-12,
                    gamma_max=1e-2,
                    enable_beta=self.args.enable_loss_anti,
                    enable_gamma=self.args.enable_loss_stable
                ).to(self.args.device)
            if not self.args.enable_loss_anti:
                self.args.loss_anti = 0
            if not self.args.enable_loss_stable:
                self.args.loss_stable = 0
        else:
            # two flow nets
            self.flow_net_e = FlowMatching(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.args.hidden_dim,
                depth=self.args.flow_depth
            )
            self.flow_net_a = FlowMatching(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.args.hidden_dim,
                depth=self.args.flow_depth
            )
            if self.args.flow_path and not self.args.finetune_ve:
                self.flow_net_e.load_state_dict(torch.load(self.args.flow_path, map_location=self.args.device))
                # frozen net e after pretrain
                for p in self.flow_net_e.parameters():
                    p.requires_grad = False
                self.opt = optim.Adam(self.flow_net_a.parameters(), lr=self.args.disc_lr)
            else:
                params = list(self.flow_net_e.parameters()) + list(self.flow_net_a.parameters())
                self.opt = optim.Adam(params, lr=self.args.disc_lr)

    def init(self, policy, args):
        super().init(policy, args)
        self.policy = policy
        self.args = args
        self.action_space = self.policy.action_space

        self._create_flow_net()
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.eps = 1e-8
        self.global_scale = 1e-3

    def _get_sampler(self, storage):
        agent_experience = storage.get_generator(None, mini_batch_size=self.expert_train_loader.batch_size)
        return self.expert_train_loader, agent_experience

    def _trans_batches(self, expert_batch, agent_batch):
        return expert_batch, agent_batch

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if not args.flow_state_norm:
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
        if not self.args.flow_state_norm:
            return state
        state = state.cpu().numpy()

        if obsfilt is not None:
            state = obsfilt(state, update=False)
        state = torch.tensor(state).to(self.args.device)
        return state

    def _trans_agent_state(self, state, other_state=None):
        if not self.args.flow_state_norm:
            if other_state is None:
                return state['raw_obs']
            return other_state['raw_obs']
        return rutils.get_def_obs(state)

    # rewards
    def _compute_flow_reward(self, storage, step, add_info):
        state = torch.FloatTensor(rutils.get_def_obs(storage.get_obs(step))).to(self.args.device)
        action = torch.FloatTensor(storage.actions[step]).to(self.args.device)
        action = rutils.get_ac_repr(self.action_space, action)

        if self.args.option == "scrf":
            logp_E = self.flow_net.log_prob(state, action, "expert")
            logp_A = self.flow_net.log_prob(state, action, "agent")
        else:
            x = torch.cat([state, action], dim=-1)
            logp_E = self.flow_net_e.log_prob(x)
            logp_A = self.flow_net_a.log_prob(x)

        r = (logp_E - logp_A).detach().cpu().numpy()
        s = torch.sigmoid(torch.tensor(r, device=state.device))
        eps = 1e-20

        if self.args.reward_type == 'norm':
            reward = (s + eps).log() - (1 - s + eps).log()
        elif self.args.reward_type == 'exp':
            exp_e = torch.exp(r)
            reward = - exp_e / (1 + exp_e)
        elif self.args.reward_type == 'raw':
            reward = torch.tensor(r)
        elif self.args.reward_type == 'clip':
            reward = torch.clamp(r, min=-20.0, max=20.0)
        elif self.args.reward_type == 'smooth':
            reward = -torch.log1p(torch.exp(-r))
        else:
            raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
        return reward

    def _get_reward(self, step, storage, add_info):
        masks = storage.masks[step]
        with torch.no_grad():
            self.flow_net.eval()
            reward = self._compute_flow_reward(storage, step, add_info)

            if self.args.drail_reward_norm:
                if self.returns is None:
                    self.returns = reward.clone()

                self.returns = self.returns * masks * self.args.gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8), {}
            else:
                return reward, {}

    def _compute_flow_loss(self, agent_batch, expert_batch, obsfilt):
        expert = expert_batch.to(self.args.device)
        agent = agent_batch.to(self.args.device)
        s_E = expert[:, :self.state_dim]
        a_E = expert[:, self.state_dim:]
        s_A = agent[:, :self.state_dim]
        a_A = agent[:, self.state_dim:]

        if self.args.option == "scrf":
            loss_E = self.flow_net.fm_loss(s_E, a_E, role="expert")
            loss_A = self.flow_net.fm_loss(s_A, a_A, role="agent")
            flow_loss = self.args.expert_loss_rate * loss_E + self.args.agent_loss_rate * loss_A

            use_anti = self.args.loss_anti > 0
            use_stable = self.args.loss_stable > 0
            if use_anti or use_stable:
                mix_s = torch.cat([s_E, s_A], dim=0)
                mix_a = torch.cat([a_E, a_A], dim=0).detach().requires_grad_(True)
                t_mix = torch.rand(mix_s.size(0), 1, device=mix_s.device)
                v_c, r = self.flow_net.net(mix_a, mix_s, t_mix)  # vector c and residual

                loss_anti = self.flow_net._hutch_div(mix_a, r, k=1).square().mean() if use_anti else None
                loss_stable = _jacobian_frobenius(mix_a, v_c + r).mean() if use_stable else None

                if not self.args.params_autotune:  # disable autotune
                    if use_anti:
                        flow_loss += self.args.loss_anti * loss_anti
                    if use_stable:
                        flow_loss += self.args.loss_stable * loss_stable

                else:
                    # fintune warmup
                    if self.gradnorm.step_count <= self.gradnorm.warmup_steps:
                        reg = 0.0
                        if self.args.enable_loss_anti and use_anti:
                            w_anti = loss_E.detach() / (loss_anti.detach() + self.eps) * self.global_scale
                            reg += w_anti * loss_anti
                            self.args.loss_anti = w_anti
                        if self.args.enable_loss_stable and use_stable:
                            w_st = loss_E.detach() / (loss_stable.detach() + self.eps) * self.global_scale
                            reg += w_st * loss_stable
                            self.args.loss_stable = w_st
                        flow_loss += reg

                    else:
                        # gradnorm
                        reg, self.args.loss_anti, self.args.loss_stable = self.gradnorm(loss_anti, loss_stable)
                        flow_loss += reg
                        self.gradnorm.update(loss_anti, loss_stable, self.params)
        else:
            loss_E = self.flow_net_e.fm_loss(s_E, a_E)
            loss_A = self.flow_net_a.fm_loss(s_A, a_A)
            flow_loss = self.args.expert_loss_rate * loss_E + self.args.agent_loss_rate * loss_A

        return flow_loss, loss_E, loss_A

    def _update_reward_func(self, storage):
        self.flow_net.train()
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
                loss, e_loss, a_loss = self._compute_flow_loss(expert_batch, agent_batch, obsfilt)

                # only update Va
                if self.args.flow_path and not self.args.finetune_ve:
                    self.opt.zero_grad()
                    a_loss.backward()
                else:
                    self.opt.zero_grad()
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.args.option == "scrf" and
                    self.flow_net.parameters() or
                    (list(self.flow_net_e.parameters()) + list(self.flow_net_a.parameters())),
                    self.args.disc_grad_pen
                )
                self.opt.step()

                log_vals['flow_loss'] += e_loss.item()
                log_vals['agent_loss'] += a_loss.item()
                self.step += 1
                count += 1

        for k in log_vals:
            log_vals[k] /= max(count, 1)

        if self.args.env_name.startswith("Sine"):
            if self.args.plot_during_train and self.step % self.args.save_interval == 0:
                log_vals["_disc_val_map"] = self.plot_disc_val_map(self.step)
                self._disc_plot_path.append(log_vals["_disc_val_map"])

                log_vals["_vector_field_map"] = self.plot_vector_field(self.step)
                self._vector_plot_path.append(log_vals["_vector_field_map"])

                log_vals["_reward_map"] = self.plot_reward_map(self.step)
                self._reward_plot_path.append(log_vals["_reward_map"])
        return log_vals

    def _compute_disc_val(self, state, action):
        state = state.to(self.args.device)
        action = action.to(self.args.device)

        if self.args.option == "scrf":
            logp_E = self.flow_net.log_prob(state, action, role="expert")
            logp_A = self.flow_net.log_prob(state, action, role="agent")
        else:
            logp_E = self.flow_net_e.log_prob(state, action)
            logp_A = self.flow_net_a.log_prob(state, action)

        return (logp_E - logp_A).detach()

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
        file_path = "./data/imgs/" + self.args.prefix + f"{self.args.option}_disc_val_map.png"
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
                reward = torch.tensor(d_val)
            elif self.args.reward_type == 'clip':
                reward = torch.clamp(d_val, min=-20.0, max=20.0)
            elif self.args.reward_type == 'smooth':
                reward = -torch.log1p(torch.exp(-d_val))
            else:
                raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
            reward = reward.view(100, 100).cpu().numpy().T

        plt.figure(figsize=(8, 5))
        plt.imshow(reward, extent=[0, 10, -2, 2], cmap="jet", origin="lower", aspect="auto")
        plt.colorbar()
        file_path = "./data/imgs/" + self.args.prefix + f"{self.args.option}_reward_map.png"
        folder = os.path.dirname(file_path)
        os.makedirs(folder, exist_ok=True)
        plt.savefig(file_path)
        return file_path

    def plot_vector_field(self, i):
        data = []
        ds = self.expert_train_loader.dataset
        if hasattr(ds, 'tensors'):
            data = torch.cat(ds.tensors, dim=1)
        else:
            for s, a in ds:
                data.append(torch.cat([torch.as_tensor(s).view(-1),
                                       torch.as_tensor(a).view(-1)], dim=0))
            data = torch.stack(data, dim=0)
        data_np = data.cpu().numpy()
        x_min, x_max = data_np[:, 0].min(), data_np[:, 0].max()
        y_min, y_max = data_np[:, 1].min(), data_np[:, 1].max()

        x_vals = np.linspace(x_min, x_max, 200)
        y_vals = np.linspace(y_min, y_max, 200)
        X, Y = np.meshgrid(x_vals, y_vals)
        grid = np.stack([X.ravel(), Y.ravel()], axis=-1)
        grid_t = torch.from_numpy(grid).float().to(self.args.device)
        pad = (self.state_dim + self.action_dim) - grid_t.shape[1]
        if pad > 0:
            grid_full = torch.cat([grid_t,
                                   torch.zeros(grid_t.size(0), pad, device=self.args.device)],
                                  dim=1)
        else:
            grid_full = grid_t
        sample_t = torch.full((grid_full.size(0), 1), 0.5, device=self.args.device)

        s_grid = grid_full[:, :self.state_dim]
        a_grid = grid_full[:, self.state_dim:]
        with torch.no_grad():
            if self.args.option == '2fs':
                v_E = self.flow_net_e.net(a_grid, s_grid, sample_t)
                v_A = self.flow_net_a.net(a_grid, s_grid, sample_t)
                fields = {
                    'v_E': (v_E.cpu().numpy(), 'Expert Flow'),
                    'v_A': (v_A.cpu().numpy(), 'Agent Flow'),
                }
            else:
                v_c, r = self.flow_net.net(a_grid, s_grid, sample_t)
                v_E = v_c + r
                v_A = v_c - r
                fields = {
                    'v_c': (v_c.cpu().numpy(), 'Core Flow'),
                    'r': (r.cpu().numpy(), 'Residual'),
                    'v_E': (v_E.cpu().numpy(), 'Expert Flow'),
                    'v_A': (v_A.cpu().numpy(), 'Agent Flow'),
                }

        n = len(fields)
        cols = 2
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        axes = axes.flatten()

        for ax, (key, (vec_np, title)) in zip(axes, fields.items()):
            U = vec_np[:, 0].reshape(X.shape)
            V = vec_np.shape[1] > 1 and vec_np[:, 1].reshape(X.shape) or np.zeros_like(U)
            mag = np.sqrt(U ** 2 + V ** 2)

            cf = ax.contourf(X, Y, mag, levels=100, cmap=CMAP, alpha=ALPHA_BG)
            ax.streamplot(X, Y, U, V, color=mag, cmap=CMAP, **PLOT_KW)
            ax.set_title(title)
            ax.set_xlabel('dim 0')
            ax.set_ylabel('dim 1')
            fig.colorbar(cf, ax=ax, label='||v||')

        for j in range(n, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f'Vector Field @ epoch {i}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_dir = os.path.join(self.args.prefix + '_vec_field')
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'vector_field_{i}.png')
        fig.savefig(file_path)
        plt.close(fig)
        return file_path

    def _save_animation(self, step):
        if not getattr(self.args, 'plot_during_train', False):
            return
        if step % self.args.save_interval == 0:
            path1 = self.plot_vector_field(step)
            self._vector_plot_path.append(path1)

            path2 = self.plot_reward_map(step)
            self._reward_plot_path.append(path2)

            path3 = self.plot_disc_val_map(step)
            self._disc_plot_path.append(path3)

    def _finalize_animation(self, input_path, out_path=None):
        if not self.args.save_animation or len(input_path) == 0:
            return None

        import os, imageio

        ext = self.args.animation_type.lower()
        if out_path is None:
            out_dir = os.path.join(self.args.prefix + '_vec_field')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'animation.{ext}')

        imgs = []
        for fn in input_path:
            try:
                imgs.append(imageio.imread(fn))
            except Exception:
                continue

        if ext == 'gif':
            imageio.mimsave(out_path, imgs, fps=self.args.animation_fps)
        else:
            imageio.mimsave(out_path, imgs, fps=self.args.animation_fps, codec='libx264')
        return out_path

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides
        #########################################
        # New args
        parser.add_argument('--n-flowril-epochs', type=int, default=1)
        parser.add_argument('--flow-state-norm', type=str2bool, default=True)
        parser.add_argument('--flow-path', type=str, default=None)
        parser.add_argument('--flow-depth', type=int, default=4)
        parser.add_argument('--disc-lr', type=float, default=1e-4)
        parser.add_argument('--disc-grad-pen', type=float, default=0.0)
        parser.add_argument('--expert-loss-rate', type=float, default=1.0)
        parser.add_argument('--agent-loss-rate', type=float, default=1.0)
        parser.add_argument('--option', type=str, default='scrf', choices=['scrf', '2fs'])  # stable coupled residual flow / two flow networks
        parser.add_argument('--hidden-dim', type=int, default=256)
        parser.add_argument('--enable-loss-anti', type=str2bool, default=True)
        parser.add_argument('--enable-loss-stable', type=str2bool, default=True)
        parser.add_argument('--params_autotune', type=str2bool, default=True)  # autotune for 2 parameters
        parser.add_argument('--loss-anti', type=float, default=1e-4)
        parser.add_argument('--loss-stable', type=float, default=1e-6)
        parser.add_argument('--flow-loss-rate', type=float, default=0.2)
        parser.add_argument('--finetune-ve', type=str2bool, default=False)
        parser.add_argument('--reward-type', type=str, default='raw', help="""
                Changes the reward computation. Does
                not change training.
                """)
        # for plot
        parser.add_argument('--plot-during-train', type=bool, default=False)
        parser.add_argument('--save-animation', type=bool, default=False)
        parser.add_argument('--save-animation', type=bool, default=False)
        parser.add_argument('--save-interval', type=int, default=100)
        parser.add_argument('--animation-type', type=str, default='gif')
        parser.add_argument('--animation-fps', type=int, default=20)

    def load_resume(self, checkpointer):
        super().load_resume(checkpointer)
        self.opt.load_state_dict(checkpointer.get_key(f'flow_{self.args.option}_opt'))
        self.flow_net.load_state_dict(checkpointer.get_key(f'flow_{self.args.option}'))

    def save(self, checkpointer):
        super().save(checkpointer)
        checkpointer.save_key(f'flow_{self.args.option}_opt', self.opt.state_dict())
        checkpointer.save_key(f'flow_{self.args.option}', self.flow_net.state_dict())

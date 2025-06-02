import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import datetime
from modril.toy.env import Environment
from modril.toy.utils import norm_state
from modril.toy.policy import PPO
from modril.toy.gail import DRAIL, GAIL, GAIL_MI, GAIL_Flow, GAIL_MBD
from modril.toy.discriminators import FFJORDDensity, FlowMatching
from modril.toy.toy_tasks import *


# --- Trainer refactored --- #
class Trainer:
    # register tasks
    # Registry for tasks
    TASK_REGISTRY = {
        # 1D
        'sine': Sine1D,
        'multi_sine': MultiSine1D,
        'gauss_sine': GaussSine1D,
        'poly': Poly1D,
        # 2D
        'gaussian_hill': GaussianHill2D,
        'mexican_hat': MexicanHat2D,
        'saddle': Saddle2D,
        'ripple': SinusoidalRipple2D,
        'bimodal_gaussian': BimodalGaussian2D,
    }

    def __init__(
            self,
            function,
            method,
            n_episode=1000,
            steps=100,
            hidden_dim=128,
            actor_lr=1e-3,
            critic_lr=1e-2,
            lmbda=0.95,
            agent_epochs=10,
            eps=0.2,
            gamma=0.98,
            lr_d=1e-3,
            **kwargs
    ):
        self.state_list = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = method
        # init task
        if function not in self.TASK_REGISTRY:
            raise ValueError(f"Unknown function {function}")
        self.task = self.TASK_REGISTRY[function]()
        self.task_name = function
        # define expert and env
        self.expert_s = torch.tensor(self.task.expert_s, dtype=torch.float32, device=self.device)
        if self.task.action_dim > 0:
            self.expert_a = torch.tensor(self.task.expert_a, dtype=torch.float32, device=self.device)
        else:
            self.expert_a = None
        self.env = getattr(self.task, 'env', None)
        # dims
        self.state_dim = self.task.state_dim
        self.action_dim = self.task.action_dim
        # config
        self.n_episode = n_episode
        self.steps = steps
        self.hidden_dim = hidden_dim
        self.lr = lr_d
        # init agent
        self.agent = PPO(
            self.state_dim,
            self.action_dim,
            hidden_dim,
            actor_lr,
            critic_lr,
            lmbda,
            agent_epochs,
            eps,
            gamma,
            self.device
        )
        # init GAIL (or variant)
        self._init_trainer(method)

    def _init_trainer(self, method, **kwargs):
        if method == 'gail':
            self.trainer = GAIL(self.agent, self.state_dim, self.action_dim, self.hidden_dim, self.lr,
                                device=self.device)
        elif method == 'drail':
            self.trainer = DRAIL(self.agent, self.state_dim, self.action_dim, disc_lr=self.lr, device=self.device)
        elif method in ('mine', 'nwj'):
            self.trainer = GAIL_MI(self.agent, self.state_dim, self.action_dim, disc_lr=self.lr, device=self.device,
                                   mode=method)
        elif method in ('ffjord', 'fm'):
            self.trainer = GAIL_Flow(self.agent, self.state_dim, self.action_dim, device=self.device, mode=method,
                                     lr=1e-3)
        elif method == 'modril':
            self.trainer = GAIL_MBD(self.agent, env=self.task.env, env_name="toy", device=self.device, steps=self.steps)
        else:
            raise ValueError(f"Unknown method {method}")

    def _pretrain_density(self, method, estimator, data, steps=3000, batch=512, lr=1e-5, clip_grad=1.0,
                          log_interval=500):
        """
        method: "ffjord" or "fm"
        """
        if not torch.is_tensor(data):
            data_t = torch.tensor(data, dtype=torch.float32, device=self.device)
        else:
            data_t = data.to(self.device).float()

        opt = torch.optim.Adam(estimator.parameters(), lr=lr)
        estimator.train()

        running_loss = 0.0
        pbar = tqdm(total=steps, desc="Pretraining...", ncols=100)
        for step in range(1, steps + 1):
            idx = torch.randint(0, data_t.size(0), (batch,), device=self.device)
            x0 = data_t[idx]
            s = x0[:, :self.state_dim]
            a = x0[:, self.state_dim:]

            if method == "ffjord":
                loss = estimator.nll(x0)
            else:  # fm
                loss = estimator.c_fm_loss(s, a, dequant_std=0.02)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), clip_grad)
            opt.step()

            running_loss += loss.item()
            if step % log_interval == 0:
                avg = running_loss / log_interval
                running_loss = 0.0
                pbar.set_postfix(avg_loss=f"{avg:.4f}")

            pbar.update(1)

        pbar.close()

        estimator.eval()
        for p in estimator.parameters():
            p.requires_grad_(False)

        print("Density_E Pretrain Done\n")
        return estimator

    def runner(self):
        # pretrain for FFJORD
        if self.method == "ffjord" or self.method == "fm":
            xs_E_full = torch.cat([self.expert_s, self.expert_a], dim=1)  # [N, 2]
            density_E = self._pretrain_density(
                self.method,
                FFJORDDensity(self.state_dim + self.action_dim).to(
                    self.device) if self.method == "ffjord" else FlowMatching(self.state_dim, self.action_dim,
                                                                              self.device).to(self.device),
                xs_E_full,
                int(self.n_episode * self.steps / 1000) if self.method == "ffjord" else self.n_episode * self.steps
            )
            self.trainer.E = density_E

        # training loop
        if self.env is None:
            print("No environment defined for this task; skipping runner.")
            return
        with tqdm(total=self.n_episode, desc='Progress') as pbar:
            for ep in range(self.n_episode):
                state = self.env.reset()
                state_list, action_list, next_state_list = [], [], []
                for step in range(self.steps):
                    action = self.agent.take_action(state)
                    next_state, reward, done, info = self.env.step(state, action)
                    # next_state, true_y = self.env.step(state, action)
                    state_list.append(state)
                    action_list.append(action)
                    next_state_list.append(next_state)
                    state = next_state
                # use numpy expert arrays for training
                self.trainer.learn(
                    self.task.expert_s,
                    self.task.expert_a,
                    state_list,
                    action_list,
                    next_state_list
                )
                pbar.update(1)
        self.state_list = state_list
        self.action_list = action_list

    def plot(self):
        s_gt = np.array(self.expert_s)  # ground‐truth states
        a_gt = np.array(self.expert_a)  # ground‐truth actions
        s_pred = np.array(self.state_list)  # pred states
        a_pred = np.array(self.action_list)  # pred actions

        if s_gt.ndim == 1:
            s_gt = s_gt.reshape(-1, 1)
        if a_gt.ndim == 1:
            a_gt = a_gt.reshape(-1, 1)
        if s_pred.ndim == 1:
            s_pred = s_pred.reshape(-1, 1)
        if a_pred.ndim == 1:
            a_pred = a_pred.reshape(-1, 1)

        dim_s = s_gt.shape[1]
        dim_a = a_gt.shape[1]

        plt.figure(figsize=(6, 6))

        if dim_s == 1:
            plt.scatter(s_gt[:, 0], a_gt[:, 0], label='Ground Truth', alpha=0.5)
            plt.scatter(s_pred[:, 0], a_pred[:, 0], label='Predicted', alpha=0.5)
            plt.xlabel('state')
            plt.ylabel('action')
            plt.title(f'1D {self.task_name} Task: state vs action ({{self.method}})')

        elif dim_s == 2:
            # 2a) action_dim=2：quiver
            if dim_a == 2:
                # ground truth
                plt.quiver(
                    s_gt[:, 0], s_gt[:, 1],
                    a_gt[:, 0], a_gt[:, 1],
                    angles='xy', scale_units='xy', scale=1,
                    color='C0', alpha=0.6, label='GT arrow'
                )
                # predicted
                plt.quiver(
                    s_pred[:, 0], s_pred[:, 1],
                    a_pred[:, 0], a_pred[:, 1],
                    angles='xy', scale_units='xy', scale=1,
                    color='C1', alpha=0.6, label='Pred arrow'
                )
                plt.xlabel('state x')
                plt.ylabel('state y')
                plt.title(f'2D {self.task_name} Task: quiver plot (action as vector)')

            # 2b) action_dim=1：scatter
            elif dim_a == 1:
                sc1 = plt.scatter(
                    s_gt[:, 0], s_gt[:, 1],
                    c=a_gt[:, 0], cmap='viridis',
                    label='GT (colored by action)', s=30, alpha=0.7
                )
                cbar1 = plt.colorbar(sc1)
                cbar1.set_label('action (GT)')

                # predicted
                sc2 = plt.scatter(
                    s_pred[:, 0], s_pred[:, 1],
                    c=a_pred[:, 0], cmap='plasma',
                    marker='x', label='Pred (colored)', s=30, alpha=0.7
                )
                cbar2 = plt.colorbar(sc2)
                cbar2.set_label('action (Pred)')

                plt.xlabel('state x')
                plt.ylabel('state y')
                plt.title(f'2D {self.task_name} Task: scatter plot ({self.method})')

            else:
                raise ValueError(f"Unsupported action dimension: {dim_a}")

        else:
            raise ValueError(f"Unsupported state dimension: {dim_s}")

        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('result.png', dpi=150)
        plt.show()

    # def plot(self, kind='reward_heatmap', **kwargs):
    #     if kind == 'reward_heatmap':
    #         self._plot_reward_heatmap(**kwargs)
    #     elif kind == 'density_ratio':
    #         self._plot_density_ratio(**kwargs)
    #     else:
    #         raise ValueError(f"Unknown plot kind {kind}")

    # def _plot_reward_heatmap(self, resolution=100, extent=None):
    #     # supports 1D and 2D
    #     if self.state_dim == 1:
    #         xs = np.linspace(extent[0], extent[1], resolution)
    #         with torch.no_grad():
    #             s = torch.tensor(xs[:, None], device=self.device).float()
    #             # reward = log pE - log ppi
    #             r = self.trainer.E.log_prob(torch.cat([s, torch.zeros_like(s)], 1))  # dummy a=0
    #             r = r.cpu().numpy()
    #         plt.plot(xs, r)
    #         plt.title('Reward Heatmap (1D)')
    #         plt.xlabel('s')
    #         plt.ylabel('r')
    #     elif self.state_dim == 2:
    #         xs = ys = np.linspace(extent[0], extent[1], resolution)
    #         grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    #         with torch.no_grad():
    #             s = torch.tensor(grid, device=self.device).float()
    #             r = self.trainer.E.log_prob(s).cpu().numpy().reshape(resolution, resolution)
    #         plt.imshow(r, extent=(*extent, *extent), origin='lower')
    #         plt.colorbar()
    #         plt.title('Reward Heatmap (2D)')
    #     else:
    #         raise NotImplementedError
    #     plt.show()
    #
    # def _plot_density_ratio(self, bins=50):
    #     # compare expert vs policy reward distributions
    #     with torch.no_grad():
    #         re = self.trainer.E.log_prob(self.expert_sa).cpu().numpy()
    #         sp = torch.randn_like(self.expert_sa)  # placeholder
    #         rp = self.trainer.E.log_prob(sp).cpu().numpy()
    #     plt.hist(re, bins=bins, alpha=0.5, label='expert')
    #     plt.hist(rp, bins=bins, alpha=0.5, label='policy')
    #     plt.legend()
    #     plt.title('Density Ratio Dist')
    #     plt.show()


if __name__ == '__main__':
    tr = Trainer('saddle', 'ffjord')
    tr.runner()
    tr.plot()
    # tr.plot(kind='reward_heatmap', extent=(-1, 1))
    # tr.plot(kind='density_ratio')

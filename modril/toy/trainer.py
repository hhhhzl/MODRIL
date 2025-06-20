import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from modril.toy.policy import PPO
from modril.toy.gail import DRAIL, GAIL, GAIL_MI, GAIL_Flow, GAIL_MBD, EnergyGAIL, GAIL_FlowShare
from modril.toy.discriminators import FFJORDDensity, FlowMatching, DEENDensity, CoupledFlowMatching, \
    _jacobian_frobenius, _hutchinson_div
from modril.toy import TASK_REGISTRY
from modril.toy.utils import create_env
from modril.toy.networks import GradNorm2D
import random
import numpy as np
from time import time

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# --- Trainer refactored --- #
class Trainer:
    def __init__(
            self,
            function,
            method,
            n_episode=1000,
            steps=100,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            hidden_dim=256,
            actor_lr=1e-3,
            critic_lr=1e-2,
            lmbda=0.95,
            agent_epochs=10,
            eps=0.2,
            gamma=0.98,
            lr_d=1e-3,
            pretrain=True,
            env_type='dynamic',
            enable_beta=True,
            enable_gamma=True,
            **kwargs
    ):
        self.state_list = None
        self.device = device
        self.method = method
        # init task
        if function not in TASK_REGISTRY:
            raise ValueError(f"Unknown function {function}")
        self.task = TASK_REGISTRY[function]()
        self.task_name = function
        # define expert and env
        self.expert_s = torch.tensor(self.task.expert_s, dtype=torch.float32, device=self.device)
        if self.task.action_dim > 0:
            self.expert_a = torch.tensor(self.task.expert_a, dtype=torch.float32, device=self.device)
            a_np = np.array(self.task.expert_a) + np.random.randn(*self.task.expert_a.shape) * 0.1
            self.expert_a = torch.tensor(a_np, dtype=torch.float32, device=self.device)
        else:
            self.expert_a = None

        self.env_type = env_type

        # dims
        self.state_dim = self.task.state_dim
        self.action_dim = self.task.action_dim
        self.pretrain = pretrain
        self.env = create_env(
            self.task_name,
            env_type,
            self.task.expert_s,
            self.task.expert_a,
            self.state_dim,
            self.action_dim,
            getattr(self.task, 'x', None)
        )

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

        self.enable_beta = enable_beta
        self.enable_gamma = enable_gamma
        self.gradnorm = GradNorm2D(
            beta_init=1e-4,
            gamma_init=1e-5,
            alpha=0.1,
            warmup_steps=100,
            update_freq=10,
            ema_decay=0.9,
            beta_min=1e-8,
            beta_max=1e1,
            gamma_min=1e-12,
            gamma_max=1e-2,
            enable_beta=self.enable_beta,
            enable_gamma=self.enable_gamma
        ).to(self.device)

        # for plot metrics
        self.reward_history = []
        self.reward_min_history = []
        self.reward_max_history = []

        self.logpE_history = []
        self.logpA_history = []
        self.kl_history = []
        self.all_states = []
        self.all_actions = []
        self.last_k_rewards = []
        self.run_time = 0

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
        elif method == 'flowril':
            self.trainer = GAIL_FlowShare(self.agent, self.state_dim, self.action_dim, device=self.device, lr=1e-3)
        elif method == 'ebil':
            self.trainer = EnergyGAIL(self.agent, self.state_dim, self.action_dim, self.hidden_dim, device=self.device)
        elif method == 'modril':
            self.trainer = GAIL_MBD(self.agent, env=self.env, state_dim=self.state_dim, action_dim=self.action_dim,
                                    env_name="toy", device=self.device, steps=self.steps)
        else:
            raise ValueError(f"Unknown method {method}")

    def _pretrain_density(
            self,
            method,
            estimator,
            data,
            steps=3000,
            batch=512,
            lr=1e-4,
            clip_grad=1.0,
            log_interval=10
    ):
        """
        method: "ffjord" or "fm" or "DEEN"
        """
        if not torch.is_tensor(data):
            data_t = torch.tensor(data, dtype=torch.float32, device=self.device)
        else:
            data_t = data.to(self.device).float()

        opt = torch.optim.Adam(estimator.parameters(), lr=lr)
        estimator.train()

        best_avg = float('inf')
        plateau = 0
        window, patience = 200, 1500
        fm_hist = []

        eps = 1e-8
        global_scale = 1e-3

        running_loss = 0.0
        pbar = tqdm(total=steps, desc=f"Pretraining {self.env_type}-{self.task_name}-{self.method}")
        params = None

        if method == "flowril":
            params = list(estimator.net.parameters())
        for step in range(1, steps + 1):
            idx = torch.randint(0, data_t.size(0), (batch,), device=self.device)
            x0 = data_t[idx]
            s = x0[:, :self.state_dim]
            a = x0[:, self.state_dim:]

            if method == "ffjord":
                loss = estimator.nll(x0)
            elif method == "fm":  # fm
                loss = estimator.c_fm_loss(s, a, dequant_std=0.02)
            elif method == "flowril":
                s_mix = s
                a_mix = a + 0.1 * torch.randn_like(a)
                loss_fm = estimator.c_fm_loss(s, a, role="expert")
                loss_A = estimator.c_fm_loss(s_mix, a_mix, role="agent")

                mix_a = a.detach().requires_grad_(True)
                t = torch.rand_like(mix_a[:, :1])
                v_c, r = estimator.net(mix_a, s_mix, t)

                if self.enable_beta:
                    loss_anti = _hutchinson_div(mix_a, r, k=1).square().mean()
                else:
                    loss_anti = torch.tensor(0.0, device=v_c.device)

                if self.enable_gamma:
                    j_pen = _jacobian_frobenius(mix_a, v_c + r).mean()
                else:
                    j_pen = torch.tensor(0.0, device=v_c.device)

                if self.gradnorm.step_count <= self.gradnorm.warmup_steps:
                    loss_reg = 0
                    if self.enable_beta:
                        weight_anti = loss_fm.detach() / (loss_anti.detach() + eps) * global_scale
                        loss_reg += weight_anti * loss_anti
                        beta = weight_anti
                    else:
                        beta = torch.tensor(0.0, device=loss_anti.device)

                    if self.enable_gamma:
                        weight_jpen = loss_fm.detach() / (j_pen.detach() + eps) * global_scale
                        loss_reg += weight_jpen * j_pen
                        gamma = weight_jpen
                    else:
                        gamma = torch.tensor(0.0, device=loss_anti.device)
                else:
                    loss_reg, beta, gamma = self.gradnorm(loss_anti, j_pen)

                loss = loss_fm + loss_A + loss_reg
                estimator.beta = beta.item()
                estimator.gamma = gamma.item()

                if self.gradnorm.step_count > self.gradnorm.warmup_steps:
                    self.gradnorm.update(loss_anti, j_pen, params)

            else:
                loss = estimator.deen_loss(x0)

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(estimator.parameters(), clip_grad)
            opt.step()

            if method == "flowril":
                fm_hist.append(loss_fm.item())
                if len(fm_hist) > window:
                    fm_hist.pop(0)
                avg = sum(fm_hist) / len(fm_hist)
                if avg + 1e-4 < best_avg:
                    best_avg, plateau = avg, 0
                else:
                    plateau += 1
                if plateau >= patience:
                    pbar.write(f"[EarlyStop] stop@step {step}, fm_avg={avg:.5f}")
                    break

            running_loss += loss.item()
            if step % log_interval == 0:
                avg = running_loss / log_interval
                running_loss = 0.0
                pbar.set_postfix(avg_loss=f"{avg:.4f}")

            pbar.update(1)

        pbar.close()
        estimator.eval()
        if method != "flowril":
            for p in estimator.parameters():
                p.requires_grad_(False)
        return estimator

    def runner(self):
        # pretrain
        if (self.pretrain and (self.method in ["ffjord", "fm", "flowril"])) or self.method == 'ebil':
            xs_E_full = torch.cat([self.expert_s, self.expert_a], dim=1)  # [N, 2]
            if self.method == "ffjord":
                density_E = self._pretrain_density(
                    self.method,
                    FFJORDDensity(self.state_dim + self.action_dim, self.device).to(self.device),
                    xs_E_full,
                    int(self.n_episode * self.steps / 100)
                )
            elif self.method == "fm":
                density_E = self._pretrain_density(
                    self.method,
                    FlowMatching(self.state_dim, self.action_dim, self.device).to(self.device),
                    xs_E_full,
                    30000
                )
            elif self.method == 'ebil':
                density_E = self._pretrain_density(
                    self.method,
                    DEENDensity(self.state_dim + self.action_dim, hidden_dim=self.hidden_dim, sigma=0.1).to(
                        self.device),
                    xs_E_full,
                    30000
                )
            elif self.method == "flowril":
                density_E = self._pretrain_density(
                    self.method,
                    CoupledFlowMatching(self.state_dim, self.action_dim).to(self.device),
                    xs_E_full,
                    10000
                )
            else:
                raise
            if self.method == 'flowril':
                self.trainer.field = density_E
                self.trainer.beta_anti = density_E.beta
                self.trainer.gamma_stab = density_E.gamma
                print(self.trainer.gamma_stab)
            else:
                self.trainer.E = density_E

        # training loop
        if self.env is None:
            print("No environment defined for this task; skipping runner.")
            return

        start = time()
        with tqdm(total=self.n_episode, desc=f'Progress {self.env_type}-{self.task_name}-{self.method}') as pbar:
            for ep in range(self.n_episode):
                state = self.env.reset()
                state_list, action_list, next_state_list = [], [], []
                env_rewards = []
                for step in range(self.steps):
                    action = self.agent.take_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    state_list.append(state)
                    action_list.append(action)
                    next_state_list.append(next_state)
                    env_rewards.append(reward)
                    state = next_state

                # use numpy expert arrays for training
                self.trainer.learn(
                    self.task.expert_s,
                    self.task.expert_a,
                    state_list,
                    action_list,
                    next_state_list
                )

                if ep >= self.n_episode - 10:
                    self.last_k_rewards.append(env_rewards)

                # for metrics plot
                rewards_np = np.array(env_rewards, dtype=float)
                avg_env_reward = float(rewards_np.mean())
                min_env_reward = float(np.percentile(rewards_np, 25))
                max_env_reward = float(np.percentile(rewards_np, 75))

                self.reward_history.append(avg_env_reward)
                self.reward_min_history.append(min_env_reward)
                self.reward_max_history.append(max_env_reward)

                if self.method in ("ffjord", "fm"):
                    xs_expert = torch.cat([self.expert_s, self.expert_a], dim=1).to(self.device)  # [N, D]
                    with torch.no_grad():
                        logp_E_expert = self.trainer.E.log_prob(xs_expert).cpu().numpy()  # shape (N,)
                        logp_A_expert = self.trainer.A.log_prob(xs_expert).cpu().numpy()  # shape (N,)

                    mean_logpE = float(np.mean(logp_E_expert))
                    mean_logpA = float(np.mean(logp_A_expert))
                    self.logpE_history.append(mean_logpE)
                    self.logpA_history.append(mean_logpA)

                    kl_ep = float(np.mean(logp_E_expert - logp_A_expert))
                    self.kl_history.append(kl_ep)
                elif self.method in "modril":
                    logp_E_expert = self.trainer.mbd.g_E.cpu().numpy()  # shape (N,)
                    logp_A_expert = self.trainer.mbd.g_A.cpu().numpy()

                    mean_logpE = float(np.mean(logp_E_expert))
                    mean_logpA = float(np.mean(logp_A_expert))
                    self.logpE_history.append(mean_logpE)
                    self.logpA_history.append(mean_logpA)
                    kl_ep = float(np.mean(logp_E_expert - logp_A_expert))
                    self.kl_history.append(kl_ep)
                elif self.method in ("gail", "drail"):
                    expert_s_t = self.expert_s
                    expert_a_t = self.expert_a

                    eps = 1e-8
                    if self.method == "gail":
                        D_expert = self.trainer.discriminator(expert_s_t, expert_a_t).clamp(eps, 1 - eps)
                    else:  # "drail"
                        xs_expert = torch.cat([expert_s_t, expert_a_t], dim=1)
                        D_expert = self.trainer.discriminator(xs_expert).clamp(eps, 1 - eps)

                    log_ratio = torch.log(D_expert) - torch.log(1 - D_expert)
                    kl_ep = float(log_ratio.mean().detach().cpu().numpy())
                    self.kl_history.append(kl_ep)
                    self.logpE_history.append(None)
                    self.logpA_history.append(None)
                else:
                    self.logpE_history.append(None)
                    self.logpA_history.append(None)
                    self.kl_history.append(None)

                self.all_states.append(state_list)
                self.all_actions.append(action_list)
                pbar.update(1)

        self.run_time = time() - start

    def plot(self, filepath=".", K=5):
        s_gt = np.array(self.expert_s.cpu())  # ground‐truth states
        a_gt = np.array(self.expert_a.cpu())  # ground‐truth actions

        last_states = self.all_states[-K:]
        last_actions = self.all_actions[-K:]

        flat_states = []
        flat_actions = []
        for ep_states, ep_actions in zip(last_states, last_actions):
            for s in ep_states:
                arrs = np.asarray(s, dtype=np.float32).reshape(-1)
                arrs = arrs.reshape(self.state_dim)
                flat_states.append(arrs)
            for a in ep_actions:
                arra = np.asarray(a, dtype=np.float32).reshape(-1)
                arra = arra.reshape(self.action_dim)
                flat_actions.append(arra)

        s_pred = np.stack(flat_states, axis=0)
        a_pred = np.stack(flat_actions, axis=0)  # pred actions

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
            plt.title(f'1D {self.env_type}-{self.task_name} Task: state vs action ({self.method})')

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
                plt.title(f'2D {self.env_type}-{self.task_name} Task: quiver plot (action as vector)')

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
                plt.title(f'2D {self.env_type}-{self.task_name} Task: scatter plot ({self.method})')

            else:
                raise ValueError(f"Unsupported action dimension: {dim_a}")

        else:
            raise ValueError(f"Unsupported state dimension: {dim_s}")

        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{filepath}/result.png', dpi=150)
        plt.close()

    def plot_metrics(self, filepath="."):
        total_eps = len(self.reward_history)
        episodes_full = np.arange(1, total_eps + 1)

        max_points = 300
        if total_eps <= max_points:
            idx_ds = np.arange(total_eps)
        else:
            idx_ds = np.linspace(0, total_eps - 1, max_points, dtype=int)

        ep_ds = episodes_full[idx_ds]

        reward_mean = np.array(self.reward_history)
        reward_min = np.array(self.reward_min_history)
        reward_max = np.array(self.reward_max_history)

        reward_mean_ds = reward_mean[idx_ds]
        reward_min_ds = reward_min[idx_ds]
        reward_max_ds = reward_max[idx_ds]

        plt.figure(figsize=(8, 4))
        plt.fill_between(
            ep_ds,
            reward_min_ds,
            reward_max_ds,
            color='C0',
            alpha=0.2,
            label='Reward Bounds (IQR)'
        )
        plt.plot(
            ep_ds,
            reward_mean_ds,
            color='C0',
            label='Avg Reward'
        )
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Reward: {self.env_type}-({self.task_name})-({self.method})')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{filepath}/rewards.png', dpi=150)
        plt.close()

        if any(v is not None for v in self.logpE_history):
            logpE_arr = np.array([v if v is not None else np.nan for v in self.logpE_history])
            logpA_arr = np.array([v if v is not None else np.nan for v in self.logpA_history])

            logpE_ds = logpE_arr[idx_ds]
            logpA_ds = logpA_arr[idx_ds]

            plt.figure(figsize=(8, 4))
            plt.plot(
                ep_ds,
                logpE_ds,
                label='mean score (expert)',
                color='C1'
            )
            plt.plot(
                ep_ds,
                logpA_ds,
                label='mean score (agent)',
                color='C2'
            )
            plt.xlabel('Episode')
            plt.ylabel('log p')
            plt.title(f'score_E vs score_A: {self.env_type}-({self.task_name})-({self.method})')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{filepath}/scores.png', dpi=150)
            plt.close()

        if any(v is not None for v in self.kl_history):
            kl_arr = np.array([v if v is not None else np.nan for v in self.kl_history])
            kl_ds = kl_arr[idx_ds]

            plt.figure(figsize=(8, 4))
            plt.plot(
                ep_ds,
                kl_ds,
                label='Discriminator Surrogate',
                color='C3'
            )
            plt.xlabel('Episode')
            plt.ylabel('Surrogate')
            plt.title(f'Surrogate: {self.env_type}-({self.task_name}) - ({self.method})')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{filepath}/surrogate_curve.png', dpi=150)
            plt.close()


if __name__ == '__main__':
    tr = Trainer(
        'sine',
        'flowril',
        env_type='static',
        pretrain=True
    )
    tr.runner()
    tr.plot(K=10)
    tr.plot_metrics()

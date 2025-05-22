import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import datetime
from modril.reward_f.env import Environment
from modril.reward_f.utils import norm_state
from modril.reward_f.policy import PPO
from modril.reward_f.gail import DRAIL, GAIL, GAIL_MI, GAIL_Flow
from modril.reward_f.discriminators import FFJORDDensity, FlowMatching


class Trainer:
    def __init__(
            self,
            function,
            method,
            lr_d=1e-3,
            hidden_dim=128,
            n_episode=1000,
            steps=100,
    ):
        self.action_list = None
        self.state_list = None
        self.return_list = None
        self.state_dim = None
        self.expert_a = None
        self.x = None
        self.expert_s = None
        self.action_dim = None
        self.env = None
        self.agent = None

        self.method = method
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.n_episode = n_episode
        self.steps = steps
        self.hidden_dim = hidden_dim

        # init function task and env
        self.init(function)

        # init agent
        actor_lr, critic_lr, gamma, lmbda, agent_epochs, eps = 1e-3, 1e-2, 0.98, 0.95, 10, 0.2
        self.agent = PPO(self.state_dim, self.hidden_dim, self.action_dim, actor_lr, critic_lr, lmbda, agent_epochs,
                         eps, gamma, self.device)

        # init trainer
        if method == 'base':
            self.trainer = GAIL(self.agent, self.state_dim, self.action_dim, self.hidden_dim, lr_d, device=self.device)
        elif method == 'drail':
            self.trainer = DRAIL(self.agent, self.state_dim, self.action_dim, disc_lr=lr_d, device=self.device)
        elif method == 'mine' or method == 'nwj':
            self.trainer = GAIL_MI(self.agent, self.state_dim, self.action_dim, disc_lr=lr_d, device=self.device,
                                   mode=method)
        elif method == 'ffjord' or method == 'fm':
            self.trainer = GAIL_Flow(self.agent, self.state_dim, self.action_dim, device=self.device, mode=method,
                                     lr=1e-3)

    def init(
            self,
            function,
            **kwargs
    ):
        if function == "sine":
            np.random.seed(42)  # Set random seed for reproducibility
            amplitude = 1  # Amplitude of the sine wave
            frequency = 0.1  # Frequency of the sine wave
            phase = 0.0  # Phase shift of the sine wave
            noise_std = 0.05  # Standard deviation of the Gaussian noise
            scale = 4
            self.x = np.linspace(0, 10, num=1000)
            self.expert_a = amplitude * np.sin(scale * frequency * np.pi * self.x + phase) + np.random.normal(0, noise_std, size=len(self.x))
            self.expert_s = norm_state(self.x)
            data_raw = np.stack([self.x, self.expert_a], axis=1)
            self.state_dim = 1
            self.action_dim = 1
            self.env = Environment(data_raw, self.x)

    def _pretrain_density(self, method, estimator, data, steps=3000, batch=512, lr=1e-5, clip_grad=0.1):
        print("Pretrain Density_E ...")
        opt = torch.optim.Adam(estimator.parameters(), lr=lr)
        with tqdm(total=steps, desc="Pretraining") as pbar:
            for step in range(steps):
                idx = np.random.choice(len(data), batch, False)
                if method == "ffjord":
                    loss = estimator.nll(data[idx])
                elif method == "fm":
                    loss = estimator.fm_loss(data[idx])
                opt.zero_grad()
                loss.backward()
                if method == "fm":
                    torch.nn.utils.clip_grad_norm_(estimator.parameters(), clip_grad)
                opt.step()
                if (step + 1) % 500 == 0:
                    print(f"[Density_E pretrain] step {step+1:5d}  loss {loss.item():.4f}")
                pbar.update(1)

        estimator.eval()
        for p in estimator.parameters():
            p.requires_grad_(False)

        print("Density_E Pretrain Done\n")
        return estimator

    def runner(self):
        # pretrain for FFJORD
        if self.method == "ffjord" or self.method == "fm":
            xs_E_full = torch.tensor(np.stack([self.expert_s, self.expert_a], 1), dtype=torch.float32, device=self.device)
            density_E = self._pretrain_density(
                self.method,
                FFJORDDensity(self.state_dim + self.action_dim).to(self.device) if self.method == "ffjord" else FlowMatching(self.state_dim + self.action_dim).to(self.device),
                xs_E_full,
                int(self.n_episode * self.steps / 1000) if self.method == "ffjord" else self.n_episode * self.steps
            )
            self.trainer.E = density_E

        with tqdm(total=self.n_episode, desc="Progress") as pbar:
            for i in range(self.n_episode):
                state = self.env.reset()
                state_list = []
                action_list = []
                next_state_list = []
                for i in range(self.steps):
                    action = self.agent.take_action(state)
                    next_state, true_y = self.env.step(state, action)
                    state_list.append(state)
                    action_list.append(action)
                    next_state_list.append(next_state)
                    state = next_state
                self.trainer.learn(self.expert_s, self.expert_a, state_list, action_list, next_state_list)
                pbar.update(1)

        self.state_list = state_list
        self.action_list = action_list

    def plot(self):
        plt.scatter(self.expert_s, self.expert_a, label='Ground Truth')
        plt.scatter(self.state_list, self.action_list, label='Predicted')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('result_sine.png')
        plt.show()


if __name__ == "__main__":
    trainer = Trainer(
        function="sine",
        method='nwj'
    )
    trainer.runner()
    trainer.plot()

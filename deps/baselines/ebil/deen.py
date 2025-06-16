import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
from modril.utils.ema import ema


def str2bool(v):
    return v.lower() == "true"


def norm_vec(x, mean, std):
    obs_x = torch.clamp(
        (x - mean) / (std + 1e-8),
        -10.0,
        10.0,
    )
    return obs_x


class DEENDensity(nn.Module):
    """
    Denoising Energy Estimator Network (DEEN)
    for estimating expert(state,action) energy E(s,a)：
       L = E_{x~ρ_E, y=x+N(0,σ^2I)} || x - y + σ^2 ∇_y E_θ(y) ||^2
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, sigma=0.1, depth=4):
        """
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Number of units in each hidden layer
            sigma (float): Noise scale for DEEN loss
            depth (int): Number of hidden layers (>=1)
        """
        super().__init__()
        self.dim = state_dim + action_dim
        self.hidden_dim = hidden_dim
        self.sigma = sigma
        self.depth = max(1, depth)

        # Build network with configurable number of layers
        layers = []
        # Input layer
        layers.append(nn.Linear(self.dim, hidden_dim))
        layers.append(nn.ReLU())
        # Additional hidden layers
        for _ in range(self.depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        # Assemble into a Sequential module
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]

    def deen_loss(self, x):
        eps = torch.randn_like(x) * self.sigma  # [B, dim]
        y = x + eps  # [B, dim]

        y.requires_grad_(True)
        Ey = self.forward(y)  # [B]
        grad_y = torch.autograd.grad(
            outputs=Ey.sum(),
            inputs=y,
            create_graph=True,
            retain_graph=True
        )[0]  # [B, dim]
        residual = x - y + (self.sigma ** 2) * grad_y  # [B, dim]
        loss = (residual ** 2).mean() * residual.shape[1]
        return loss

    def log_energy(self, x):
        with torch.no_grad():
            return self.forward(x)  # [B]


if __name__ == "__main__":
    # Create the environment
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-load-path', type=str, default='expert_datasets/maze.pt')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sigma', type=float, default=1e-1)
    parser.add_argument('--num-epoch', type=int, default=8000)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--save-path', type=str, default='data/pre')
    args = parser.parse_args()
    task = "ebm"

    print("=======================================================")
    print(f"Task = {task}")
    print(f"Hidden dimension = {args.hidden_dim}")
    print(f"Depth = {args.depth}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    num_epoch = args.num_epoch

    env = args.traj_load_path.split('/')[-1][:-3]
    model_save_path = f'{args.save_path}/{env}/{task}/trained_models'
    image_save_path = f'{args.save_path}/{env}/{task}/trained_imgs'

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, exist_ok=True)

    data = torch.load(args.traj_load_path)
    # Demonstration normalization
    obs = data["obs"]
    print("obs shape", obs.shape)
    if args.norm:
        obs_mean = obs.mean(0)
        obs_std = obs.std(0)
        print(f"obs std: {obs_std}")
        obs = norm_vec(obs, obs_mean, obs_std)

    actions = data["actions"]
    print("actions shape", actions.shape)
    if args.norm:
        actions_mean = actions.mean(0)
        actions_std = actions.std(0)
        print(f"actions std: {actions_std}")
        actions = norm_vec(actions, actions_mean, actions_std)

    dataset = torch.cat((obs, actions), 1)
    sample_num = dataset.size()[0]
    if sample_num % 2 == 1:
        dataset = dataset[1:sample_num, :]

    print("after", dataset.size())
    print("actions.dtype:", actions.dtype)
    print("Training model...")

    dataset = dataset.float()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # output dimension is state_dim + action_dim，inputs are x and step
    estimator = DEENDensity(
        state_dim=obs.shape[1],
        action_dim=actions.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        sigma=args.sigma
    ).to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=args.lr)
    estimator.train()

    train_loss_list = []
    running_loss = 0.0
    for t in tqdm(range(1, num_epoch + 1)):
        total_loss = 0
        for idx, batch_x in enumerate(dataloader):
            loss = estimator.deen_loss(batch_x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), 1)
            optimizer.step()
            loss = loss.cpu().detach()
            total_loss += loss
        ave_loss = total_loss / len(dataloader)
        train_loss_list.append(ave_loss)

        if t % 200 == 0:
            data_np = dataset.cpu().numpy()
            x_min, x_max = data_np[:, 0].min(), data_np[:, 0].max()
            y_min, y_max = data_np[:, 1].min(), data_np[:, 1].max()
            x_vals = np.linspace(x_min, x_max, 200)
            y_vals = np.linspace(y_min, y_max, 200)
            X, Y = np.meshgrid(x_vals, y_vals)

            grid = np.stack([X.ravel(), Y.ravel()], axis=-1)
            grid_xy = torch.from_numpy(grid).float().to(device)
            pad = torch.zeros(grid_xy.size(0), dataset.shape[1] - 2, device=device)
            grid_full = torch.cat([grid_xy, pad], dim=1)

            with torch.no_grad():
                E = estimator.log_energy(grid_full)
            E = E.cpu().numpy().reshape(X.shape)

            fig, ax = plt.subplots(figsize=(6, 6))
            cf = ax.contourf(X, Y, E, levels=100, cmap='viridis')
            plt.colorbar(cf, ax=ax, label='Energy')
            ax.set_title(f'Energy Field at epoch {t}')
            ax.set_xlabel('dim0')
            ax.set_ylabel('dim1')
            ax.scatter(data_np[:2000, 0], data_np[:2000, 1], s=2, c='white', alpha=0.6, edgecolors='none')
            plt.savefig(f'{image_save_path}/{env}_energy_epoch_{t}.png')
            plt.close()

        if t % 200 == 0:
            train_iteration_list = list(range(len(train_loss_list)))
            smoothed_ema = ema(train_loss_list, 0.05)
            plt.figure(figsize=(6, 4))
            plt.plot(train_iteration_list, train_loss_list, alpha=0.3, label='Origin Loss')
            plt.plot(train_iteration_list, smoothed_ema, label=f'EMA (α={0.05})', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(env + '_deen_loss (EMA)')
            plt.legend()
            plt.savefig(f'{image_save_path}/{env}_deen_loss.png')
            plt.close()

        if t % 200 == 0:
            torch.save(estimator.state_dict(), f'{model_save_path}/{env}_deen_{t}.pt')

    torch.save(estimator.state_dict(), f'{model_save_path}/{env}_deen.pt')
    torch.save(train_loss_list, f'{model_save_path}/{env}_train_loss.pt')

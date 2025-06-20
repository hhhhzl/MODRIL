from modril.flowril.networks import SharedVNet, ConditionalVNet, GradNorm2D
import torch.nn.functional as F
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
from modril.utils.ema import ema

_RNG = torch.Generator()

CMAP = 'viridis'

PLOT_KW = dict(density=1.2, linewidth=1, arrowstyle='-|>')
ALPHA_BG = 0.85


def plot_flow(ax, X, Y, U, V, mag, title, label):
    cf = ax.contourf(X, Y, mag, levels=100, cmap=CMAP, alpha=ALPHA_BG)
    ax.streamplot(X, Y, U, V, color=mag, cmap=CMAP, **PLOT_KW)
    ax.set_title(title)
    ax.set_xlabel('z')
    ax.set_ylabel('t')
    return cf


def str2bool(v):
    return v.lower() == "true"


def norm_vec(x, mean, std):
    obs_x = torch.clamp(
        (x - mean) / (std + 1e-8),
        -10.0,
        10.0,
    )
    return obs_x


def _rademacher(shape_or_tensor, device=None):
    """Generate Rademacher (±1) noise matching *shape* or *tensor*."""
    if isinstance(shape_or_tensor, torch.Tensor):
        device = shape_or_tensor.device if device is None else device
        shape = tuple(shape_or_tensor.shape)
    else:  # shape tuple / torch.Size
        shape = shape_or_tensor
        assert device is not None, "device must be specified when passing shape"

    rng = torch.Generator(device=device)
    return torch.randint(0, 2, shape, device=device, generator=rng).float().mul_(2).sub_(1)


def _jacobian_frobenius(x: torch.Tensor, f: torch.Tensor):
    """‖∇_x f‖_F² via one Hutchinson vector (cheapest)."""
    v = _rademacher(x, x.device)
    (jv,) = torch.autograd.grad(f, x, v, create_graph=True, retain_graph=True, only_inputs=True)
    return (jv.pow(2)).sum(-1)


class FlowMatching(nn.Module):
    """
    Ho & Salimans 2023: flow matching
    goal:  E_{t∼U(0,1)} [ || vθ(x_t,t) - v∗(x_t,t) ||² ]
    x_t = (1-t)·x + t·x̃  ，x̃~N(0,I)
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int,
            depth: int = 4,
            eps: float = 1e-3,
    ):
        super().__init__()
        self.state_dim, self.action_dim = state_dim, action_dim
        self.net = ConditionalVNet(self.state_dim, self.action_dim, hidden_dim, depth)  # single flow network
        self.dim = self.state_dim + self.action_dim
        self.prior = torch.distributions.Normal(0, 1)
        self.eps = eps
        self.T = 1.0

    def _v_star(self, a_t, a0, t, noise):
        return noise - a0

    @staticmethod
    def _hutch_div(y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        v = torch.randint_like(y, 0, 2).float().mul_(2).sub_(1)  # Rademacher ±1
        (Jv,) = torch.autograd.grad(f, y, v, create_graph=True, retain_graph=True, only_inputs=True)  # J^T v
        return (Jv * v).sum(-1)

    def fm_loss(self, s, a0, dequant_std=0.02):
        """
        """
        B = s.size(0)
        a0 = a0 + torch.randn_like(a0) * dequant_std
        t = torch.rand(B, 1, device=a0.device) * (1 - 2 * self.eps) + self.eps

        noise = self.prior.sample((B, self.action_dim)).to(a0)
        a_t = (1 - t) * a0 + t * noise
        v_star = self._v_star(a_t, a0, t, noise)

        v_pred = self.net(a_t, s, t)
        weight = (1 - t).pow(1.5)  # [B,1]
        # weight = (1 - t) ** gamma + λt * t ** delta
        mse_i = ((v_pred - v_star) ** 2).sum(dim=1, keepdim=True)  # [B,1]
        return (weight * mse_i).mean()

    # ------ estimate log-ratio via path integral ------
    def log_prob(self, x: torch.Tensor, n_steps: int = 32) -> torch.Tensor:
        s, a0 = x.split([self.state_dim, x.size(-1) - self.state_dim], dim=-1)
        B = a0.size(0)
        t_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=x.device)
        delta = 1.0 / n_steps

        log_det = torch.zeros(B, device=x.device)
        a_t = a0.detach()

        for k in range(n_steps):
            t_mid = 0.5 * (t_grid[k] + t_grid[k + 1])

            with torch.enable_grad():
                a_t = a_t.detach().requires_grad_(True)
                v_t = self.net(a_t, s, t_mid.expand(B, 1))  # [B, dim_a]
                div = self._hutch_div(a_t, v_t)  # [B]

            log_det -= div * delta
            a_t = (a_t + v_t.detach() * delta)  # Euler

        # logp
        logp_prior = self.prior.log_prob(a_t).sum(-1)  # [B]
        return logp_prior + log_det


class CoupledResidualFM(nn.Module):
    """v_E = v_c + r_φ,
    v_π = v_c − r_φ
    (anti‑symmetric residual).
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int,
            depth: int = 4,
            eps: float = 1e-3
    ):
        super().__init__()
        self.s_dim, self.a_dim, self.eps = state_dim, action_dim, eps
        self.net = SharedVNet(state_dim, action_dim, hidden=hidden_dim, depth=depth)  # two heads, vector_c + residual
        self.prior = torch.distributions.Normal(0., 1.)

    def v_field(self, a_t: torch.Tensor, s: torch.Tensor, t: torch.Tensor, role: str):
        v_c, r = self.net(a_t, s, t)
        return v_c + r if role == "expert" else v_c - r

    def _v_star(self, a_t, a0, t, noise):
        return noise - a0

    def _hutch_div(self, y: torch.Tensor, f: torch.Tensor, k: int = 1) -> torch.Tensor:
        """ Hutchinson-trace estimator ∇·f with k Rademacher vectors """
        B, D = y.shape
        out = 0.0
        for _ in range(k):
            v = _rademacher((B, D), device=y.device)
            (jv,) = torch.autograd.grad(
                outputs=f,
                inputs=y,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,
            )
            out = out + (jv * v).sum(dim=-1)
        return out / k

    def fm_loss(self, s, a0, role):
        B = a0.size(0)
        device = a0.device
        t = torch.rand(B, 1, device=device) * (1 - 2 * self.eps) + self.eps
        noise = self.prior.sample((B, self.a_dim)).to(device)
        a_t = (1 - t) * a0 + t * noise
        v_star = self._v_star(a_t, a0, t, noise)
        v_pred = self.v_field(a_t, s, t, role)

        weight = (1 - t).pow(1.5)  # [B,1]
        # weight = (1 - t) ** gamma + λt * t ** delta
        return (weight * F.mse_loss(v_pred, v_star, reduction="none").sum(1, keepdim=True)).mean()

    def log_prob(
            self,
            s: torch.Tensor,
            a0: torch.Tensor,
            role: str,
            n_steps: int = 32
    ) -> torch.Tensor:
        B = a0.size(0)
        device = a0.device
        t_grid = torch.linspace(0., 1., n_steps + 1, device=device)
        delta = 1.0 / n_steps
        log_det = torch.zeros(B, device=device)
        a_t = a0.detach()
        for k in range(n_steps):
            t_mid = 0.5 * (t_grid[k] + t_grid[k + 1])
            a_t = a_t.detach().requires_grad_(True)
            v_t = self.v_field(a_t, s, t_mid.expand(B, 1), role)
            div = self._hutch_div(a_t, v_t)
            log_det -= delta * div
            a_t = a_t + v_t.detach() * delta

        logp_prior = self.prior.log_prob(a_t).sum(-1)
        return logp_prior + log_det


if __name__ == "__main__":
    # Create the environment
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-load-path', type=str, default='modril/expert_datasets/sine.pt')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eps', type=float, default=1e-2)
    parser.add_argument('--option', type=str, default='scrf',
                        choices=['scrf', '2fs'])  # stable coupled residual flow / two flow networks
    parser.add_argument('--num-epoch', type=int, default=8000)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--save-path', type=str, default='data/pre')

    # parameters for coupled residual vector loss
    parser.add_argument('--enable-loss-anti', type=str2bool, default=True)
    parser.add_argument('--enable-loss-stable', type=str2bool, default=True)
    parser.add_argument('--autotune', type=str2bool, default=True) # autotune for 2 parameters
    parser.add_argument('--loss-anti', type=float, default=1e-4)
    parser.add_argument('--loss-stable', type=float, default=1e-6)
    args = parser.parse_args()

    task = args.option
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    num_epoch = args.num_epoch

    if not args.enable_loss_anti and args.enable_loss_stable:
        task = f"{task}_no_a"
    if not args.enable_loss_stable and args.enable_loss_anti:
        task = f"{task}_no_s"
    if not args.enable_loss_anti and not args.enable_loss_stable:
        task = f"{task}_no_as"

    print("=======================================================")
    print(f"Task = {task}")
    print(f"Hidden dimension = {args.hidden_dim}")
    print(f"Depth = {args.depth}")

    env = args.traj_load_path.split('/')[-1][:-3]
    if env.lower() == "sine":
        args.norm = False
        
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    s_dim = obs.shape[1]
    a_dim = actions.shape[1]

    # output dimension is state_dim + action_dim，inputs are x and step
    gradnorm, params = None, None
    eps = 1e-8
    global_scale = 1e-3

    # for best step
    best_avg = float('inf')
    plateau = 0
    window, patience = 200, 1500
    fm_hist = []
    plot = False
    check_best = True

    if args.option == 'scrf':
        estimator = CoupledResidualFM(
            state_dim=s_dim,
            action_dim=a_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            eps=args.eps
        ).to(device)
        optimizer = torch.optim.Adam(estimator.net.parameters(), lr=args.lr)
        params = list(estimator.net.parameters())
        if args.autotune:
            gradnorm = GradNorm2D(
                beta_init=args.loss_anti,
                gamma_init=args.loss_stable,
                alpha=0.1,
                warmup_steps=100,
                update_freq=10,
                ema_decay=0.9,
                beta_min=1e-8,
                beta_max=1e1,
                gamma_min=1e-12,
                gamma_max=1e-2,
                enable_beta=args.enable_loss_anti,
                enable_gamma=args.enable_loss_stable
            ).to(device)
        if not args.enable_loss_anti:
            args.loss_anti = 0
        if not args.enable_loss_stable:
            args.loss_stable = 0

    elif args.option == '2fs':
        estimator = FlowMatching(
            state_dim=s_dim,
            action_dim=a_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            eps=args.eps
        ).to(device)
        optimizer = torch.optim.Adam(estimator.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    estimator.train()

    train_loss_list = []
    running_loss = 0.0
    for t in tqdm(range(1, num_epoch + 1)):
        total_loss = 0
        for idx, batch_x in enumerate(dataloader):
            batch_x = batch_x.to(device)
            s = batch_x[:, :s_dim]
            a = batch_x[:, s_dim:]

            if args.option == "scrf":
                loss_expert = estimator.fm_loss(s, a, role="expert")
                noise = 0.1 * torch.randn_like(a)
                a_noise = a + noise
                loss_agent = estimator.fm_loss(s, a_noise, role="agent")
                loss = loss_expert + loss_agent

                use_anti = args.loss_anti > 0
                use_stable = args.loss_stable > 0
                if use_anti or use_stable:
                    mix_a = a_noise.detach().requires_grad_(True)
                    t_noise = torch.rand_like(mix_a[..., :1])
                    v_c, r = estimator.net(mix_a, s, t_noise)

                    loss_anti = estimator._hutch_div(mix_a, r, k=1).square().mean() if use_anti else None
                    loss_stable = _jacobian_frobenius(mix_a, v_c + r).mean() if use_stable else None

                    if not args.autotune: # disable autotune
                        if use_anti:
                            loss += args.loss_anti * loss_anti
                        if use_stable:
                            loss += args.loss_stable * loss_stable

                    else:
                        # fintune warmup
                        if gradnorm.step_count <= gradnorm.warmup_steps:
                            reg = 0.0
                            if args.enable_loss_anti and use_anti:
                                w_anti = loss_expert.detach() / (loss_anti.detach() + eps) * global_scale
                                reg += w_anti * loss_anti
                                args.loss_anti = w_anti
                            if args.enable_loss_stable and use_stable:
                                w_st = loss_expert.detach() / (loss_stable.detach() + eps) * global_scale
                                reg += w_st * loss_stable
                                args.loss_stable = w_st
                            loss = loss + reg

                        else:
                            # gradnorm
                            reg, new_w_anti, new_w_st = gradnorm(loss_anti, loss_stable)
                            loss += reg
                            gradnorm.update(loss_anti, loss_stable, params)
                            args.loss_anti, args.loss_stable = new_w_anti, new_w_st

                        estimator.loss_anti = args.loss_anti
                        estimator.loss_stable = args.loss_stable

            else:
                loss = estimator.fm_loss(s, a)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), 1)
            optimizer.step()
            total_loss += loss.item()

            if args.option == "scrf":
                fm_hist.append(loss_expert.item())
            else:
                fm_hist.append(loss.item())

            if check_best:
                if len(fm_hist) > window:
                    fm_hist.pop(0)
                avg = sum(fm_hist) / len(fm_hist)
                if avg + 1e-5 < best_avg:
                    best_avg, plateau = avg, 0
                else:
                    plateau += 1
                if plateau >= patience:
                    plot = True
                    check_best = False

        ave_loss = total_loss / len(dataloader)
        train_loss_list.append(ave_loss)

        if t % 200 == 0 or plot:
            # --- prepare grid for vector field plotting ---
            data_np = dataset.cpu().numpy()
            x_min, x_max = data_np[:, 0].min(), data_np[:, 0].max()
            y_min, y_max = data_np[:, 1].min(), data_np[:, 1].max()
            x_vals = np.linspace(x_min, x_max, 200)
            y_vals = np.linspace(y_min, y_max, 200)
            X, Y = np.meshgrid(x_vals, y_vals)
            grid = np.stack([X.ravel(), Y.ravel()], axis=-1)
            grid_xy = torch.from_numpy(grid).float().to(device)
            pad = torch.zeros(grid_xy.size(0), s_dim + a_dim - 2, device=device)
            grid_full = torch.cat([grid_xy, pad], dim=1)
            sample_t = torch.full((grid_full.size(0), 1), 0.5, device=device)

            if args.option == '2fs':
                with torch.no_grad():
                    s_grid = grid_full[:, :s_dim]
                    a_grid = grid_full[:, s_dim:]
                    v_pred = estimator.net(a_grid, s_grid, sample_t)
                U = v_pred[:, 0].cpu().numpy().reshape(X.shape)
                V = v_pred[:, 1].cpu().numpy().reshape(X.shape) if a_dim > 1 else np.zeros_like(U)
                magnitude = np.sqrt(U ** 2 + V ** 2)
                plt.figure(figsize=(6, 6))
                contour = plt.contourf(X, Y, magnitude, levels=100, cmap=CMAP, alpha=ALPHA_BG)
                plt.colorbar(contour, label='||v||')
                plt.streamplot(X, Y, U, V, color=magnitude, cmap=CMAP, **PLOT_KW)
                plt.title(f'{env}_{task}_vector_field_step_{t}')
                plt.savefig(f'{image_save_path}/{env}_{task}_vector_field_{t}{"_b" if plot else ""}.png')
                plt.close()
            elif args.option == 'scrf':
                with torch.no_grad():
                    s_grid = grid_full[:, :s_dim]
                    a_grid = grid_full[:, s_dim:]
                    v_c_grid, r_grid = estimator.net(a_grid, s_grid, sample_t)

                Uc = v_c_grid[:, 0].cpu().numpy().reshape(X.shape)
                Vc = v_c_grid[:, 1].cpu().numpy().reshape(X.shape) if a_dim > 1 else np.zeros_like(Uc)
                Ur = r_grid[:, 0].cpu().numpy().reshape(X.shape)
                Vr = r_grid[:, 1].cpu().numpy().reshape(X.shape) if a_dim > 1 else np.zeros_like(Ur)

                mag_c = np.sqrt(Uc ** 2 + Vc ** 2)
                mag_r = np.sqrt(Ur ** 2 + Vr ** 2)

                # v_e = vc + r
                v_e = (v_c_grid + r_grid).cpu().numpy()
                if v_e.shape[1] == 1:
                    Ue = v_e[:, 0].reshape(X.shape)
                    Ve = np.zeros_like(Ue)
                else:
                    Ue = v_e[:, 0].reshape(X.shape)
                    Ve = v_e[:, 1].reshape(X.shape)
                mag_e = np.sqrt(Ue ** 2 + Ve ** 2)

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
                cf1 = plot_flow(ax1, X, Y, Uc, Vc, mag_c, 'Vector c', '||v_c||')
                cf2 = plot_flow(ax2, X, Y, Ur, Vr, mag_r, 'Residual r', '||r||')
                cf3 = plot_flow(ax3, X, Y, Ue, Ve, mag_e, 'Vector E', '||v_E||')
                for ax, cf, lbl in zip((ax1, ax2, ax3), (cf1, cf2, cf3), ('||v_c||', '||r||', '||v_E||')):
                    fig.colorbar(cf, ax=ax, label=lbl)

                fig.suptitle(f'{env}_{task}_vector_field_step_{t}')
                fig.savefig(f'{image_save_path}/{env}_{task}_vector_field_{t}{"_b" if plot else ""}.png')
                plt.close(fig)

        if t % 200 == 0 or plot:
            train_iteration_list = list(range(len(train_loss_list)))
            smoothed_ema = ema(train_loss_list, 0.05)
            plt.figure(figsize=(6, 4))
            plt.plot(train_iteration_list, train_loss_list, alpha=0.3, label='Origin Loss')
            plt.plot(train_iteration_list, smoothed_ema, label=f'EMA (α={0.05})', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(env + '_flow_matching_loss (EMA)')
            plt.legend()
            plt.savefig(f'{image_save_path}/{env}_fm_loss{"_b" if plot else ""}.png')
            plt.close()

        if t % 200 == 0 or plot:
            torch.save(estimator.state_dict(), f'{model_save_path}/{env}_fm_{t}{"_b" if plot else ""}.pt')

        plot = False

    torch.save(estimator.state_dict(), f'{model_save_path}/{env}_fm.pt')
    torch.save(train_loss_list, f'{model_save_path}/{env}_train_loss.pt')

import torch
import numpy as np
from tqdm import tqdm
from modril.toy.utils import dynamic_convert


class MBDScore:
    def __init__(
            self,
            env,
            env_name,
            steps,
            state_dim,
            action_dim,
            seed=0,
            disable_recommended_params=False,
            device='cpu',
            temp_sample: float = 0.5,
            num_diffusion_steps: int = 1000,
    ):
        """
        Initialize the diffusion-based trajectory optimizer with PyTorch.

        Args:
            env: The environment object
            env_name: Name of the environment
            seed: Random seed for reproducibility
            enable_demo: Whether to use demonstration trajectories
            disable_recommended_params: Whether to disable recommended parameters
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device
        self.env = env
        self.env_name = env_name
        self.disable_recommended_params = disable_recommended_params

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Default parameters
        self.Nsample = 200
        self.Hsample = steps
        self.num_diffusion_steps = num_diffusion_steps
        self.alphas = torch.linspace(1.0, 1e-3, num_diffusion_steps + 1, device=self.device)

        self.temp_sample = temp_sample
        self.beta0 = 5e-6
        self.betaT = 1e-3
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Recommended parameters for specific environments
        self.recommended_params = {
            "temp_sample": {"toy": 0.1},
            "num_diffusion_steps": {"toy": 100},
            "Nsample": {"toy": 100},
            "Hsample": {"toy": steps}
        }

        # Initialize environment parameters
        self._setup_environment()
        self._setup_diffusion_params()

        self.g_E = None
        self.g_E_mean = None
        self.g_A = None

    def _setup_environment(self):
        """Initialize the environment and related parameters."""
        # Apply recommended parameters if not disabled
        if not self.disable_recommended_params:
            for param_name in ["temp_sample", "num_diffusion_steps", "Nsample", "Hsample"]:
                recommended_value = self.recommended_params[param_name].get(
                    self.env_name, getattr(self, param_name)
                )
                setattr(self, param_name, recommended_value)
            print(f"Using recommended parameters: temp_sample={self.temp_sample}")

        self.Nx = self.env.state_dim
        self.Nu = self.env.action_dim

        # Reset environment to initial state
        self.state_init = self.env.reset()

    def _setup_diffusion_params(self):
        """Calculate diffusion process parameters."""
        # Create tensors on the specified device
        self.betas = torch.linspace(self.beta0, self.betaT, self.num_diffusion_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(1 - self.alphas_bar)

        # Conditional distribution parameters
        shifted_alphas_bar = torch.roll(self.alphas_bar, 1)
        shifted_alphas_bar[0] = 1.0  # Handle first element
        Sigmas_cond = ((1 - self.alphas) * (1 - torch.sqrt(shifted_alphas_bar)) / (1 - self.alphas_bar))
        self.sigmas_cond = torch.sqrt(Sigmas_cond)
        self.sigmas_cond[0] = 0.0

    def _foward_diffusion(self, Y0: torch.Tensor) -> torch.Tensor:
        y = Y0.clone()
        for i in range(self.num_diffusion_steps):
            eps = torch.randn_like(y)
            y = torch.sqrt(self.alphas[i]) * y + torch.sqrt(1.0 - self.alphas[i]) * eps  # Y_i
        # alpha_bar_N = self.alphas_bar[self.num_diffusion_steps - 1]  # \bar α_N
        # y_bar_N = y / torch.sqrt(alpha_bar_N)
        return y

    def _reverse_diffusion_step(self, i, Ybar_i):
        """
        Single step of the reverse diffusion process.

        Args:
            i: Current diffusion step index
            Ybar_i: Current trajectory estimate

        Returns:
            Updated trajectory estimate and mean reward
        """
        # Recover noisy trajectory
        Yi = Ybar_i * torch.sqrt(self.alphas_bar[i])

        # Sample from q_i
        H = Ybar_i.shape[0]
        eps_u = torch.randn((self.Nsample, H, self.Nu), device=self.device, dtype=Ybar_i.dtype)
        Y0s = eps_u * self.sigmas[i] + Ybar_i
        Y0s = torch.clamp(Y0s, -1.0, 1.0)

        # Evaluate sampled trajectories
        # rewss = []
        # for j in range(self.Nsample):
        #     # Convert to numpy for environment compatibility
        #     actions = Y0s[j].cpu().numpy() if self.device != 'cpu' else Y0s[j].numpy()
        #     rews, q = self._rollout(self.state_init, actions)
        #     rewss.append(rews)

        actions = Y0s[..., self.state_dim:]
        if isinstance(self.state_init, np.ndarray):
            states0 = np.repeat(self.state_init[None, ...], self.Nsample, axis=0)
        else:
            states0 = np.array([self.state_init] * self.Nsample, dtype=np.float32)

        rewss, states = self._rollout_batch(states0, actions)
        rews = torch.tensor(np.mean(rewss, axis=-1), device=self.device)
        rew_std = rews.std()
        rew_std = torch.where(rew_std < 1e-4, torch.tensor(1.0, device=self.device), rew_std)
        rew_mean = rews.mean()
        logp0 = (rews - rew_mean) / rew_std / self.temp_sample

        # Update trajectory using weighted average
        weights = torch.nn.functional.softmax(logp0, dim=0)

        # write rollout states back
        Y0s[..., :self.state_dim] = states.unsqueeze(-1)

        Y0s = Y0s.to(weights.dtype)
        Ybar = torch.einsum("n,nij->ij", weights, Y0s)

        # Compute score function and update
        score = 1 / (1.0 - self.alphas_bar[i]) * (-Yi + torch.sqrt(self.alphas_bar[i]) * Ybar)
        Yim1 = 1 / torch.sqrt(self.alphas[i]) * (Yi + (1.0 - self.alphas_bar[i]) * score)
        # Yim1 = torch.clamp(Yim1, -1.0, 1.0)
        Ybar_im1 = Yim1 / torch.sqrt(self.alphas_bar[i - 1])

        # if i % 50 == 0:
        #     print(f"[Step {i}] rews: min {rews.min().item():.4f}, max {rews.max().item():.4f}, mean {rew_mean:.4f}, std {rews.std().item():.4f}")
        #     print(f"[Step {i}] logp0 distribution: min {logp0.min().item():.4f}, max {logp0.max().item():.4f}")
        #     print(f"[Step {i}] weight: min {weights.min().item():.4f}, max {weights.max().item():.4f}, sum {weights.sum().item():.4f}")
        #     print(f"[Step {i}] Y0s range: min {Y0s.min().item():.4f}, max {Y0s.max().item():.4f}, mean {Y0s.mean().item():.4f}")
        #     print(f"[Step {i}] Ybar range: min {Ybar.min().item():.4f}, max {Ybar.max().item():.4f}")
        #     print(f"[Step {i}] score range: min {score.min().item():.4f}, max {score.max().item():.4f}")
        return score, Yim1, Ybar_im1, rews.mean().item()

    def _rollout(self, state, actions):
        """Rollout trajectory given initial state and actions."""
        rews = []
        states = [state]
        for t in range(actions.shape[0]):
            state, rew, done, _ = self.env.step(actions[t])
            rews.append(rew)
            states.append(state)
            if done:
                break
        return np.array(rews), states

    def _rollout_batch(self, states0, actions):
        B, T = actions.shape[:2]
        rews = np.zeros((B, T), dtype=np.float32)
        if np.isscalar(states0):
            states0 = np.array([states0] * B, dtype=np.float32)
        else:
            states0 = np.asarray(states0).reshape(B, )

        states_list = []
        cur_states = states0.copy()
        states_list.append(cur_states.copy())

        done_mask = np.zeros(B, dtype=bool)

        for t in range(T - 1):
            a_t = actions[:, t]
            if hasattr(self.env, "batch_step"):
                next_states, r_t, done_t, _ = self.env.batch_step(cur_states, a_t)
                next_states = np.asarray(next_states).reshape(B, )
            else:
                next_states = []
                r_t = np.zeros(B, dtype=np.float32)
                done_t = np.zeros(B, dtype=bool)
                for b in range(B):
                    if done_mask[b]:
                        next_states.append(cur_states[b])
                        continue
                    ns, rb, db, _ = self.env.step(cur_states[b], a_t[b])
                    next_states.append(ns)
                    r_t[b] = rb
                    done_t[b] = db
                next_states = np.asarray(next_states).reshape(B, )

            rews[~done_mask, t] = r_t[~done_mask]
            done_mask |= done_t
            cur_states = next_states
            states_list.append(cur_states.copy())

            if done_mask.all():
                for _ in range(T - t - 1):
                    states_list.append(cur_states.copy())
                break

        states_arr = np.stack(states_list, axis=0)  # (T, B)
        states_arr = states_arr.T  # (B, T)
        states_tensor = torch.tensor(states_arr, dtype=torch.float32)
        return rews, states_tensor

    def compute_reward(self, Ye, Ya):
        """
        Occupancy log-ratio
        :param Ye: expert state-action batch  [B, D]
        :param Ya: agent  state-action batch  [B, D]
        :return:
        rewards
        """
        Ye = torch.as_tensor(Ye, dtype=torch.float32, device=self.device)
        Ya = torch.as_tensor(Ya, dtype=torch.float32, device=self.device)

        if Ye.dim() == 1:
            Ye = Ye.unsqueeze(1)
        if Ya.dim() == 1:
            Ya = Ya.unsqueeze(1)

        with torch.no_grad():
            self.g_E = self._logp_change_reward(Ye)
            self.g_A = self._logp_change_reward(Ya)
        return self.g_E - self.g_A

        # if self.g_E is None:
        # if self.g_E_mean is None:
        # self.g_E_mean = self.g_E.mean()

    def _logp_change_reward(self, Y0):
        """
        reward-weighted reverse steps estimation
        """
        # Ybar = Y0.clone()
        Ybar = self._foward_diffusion(Y0)
        g = torch.zeros(Y0.shape[0], device=self.device)
        for i in reversed(range(0, self.num_diffusion_steps)):
            score, Y_i_minus1, Ybar_im1, _ = self._reverse_diffusion_step(i, Ybar)
            Y_i = Ybar * torch.sqrt(self.alphas_bar[i])
            delta_i = Y_i_minus1 - torch.sqrt(self.alphas[i]) * Y_i
            g += (score * delta_i).sum(dim=-1)
            Ybar = Ybar_im1
        return g

import torch
import numpy as np
from tqdm import tqdm


class MBD:
    def __init__(
            self,
            env,
            env_name,
            seed=0,
            enable_demo=False,
            disable_recommended_params=False,
            device='cpu'
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
        self.enable_demo = enable_demo
        self.disable_recommended_params = disable_recommended_params

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Default parameters
        self.temp_sample = 0.1
        self.Ndiffuse = 100
        self.Nsample = 8192
        self.Hsample = 40
        self.beta0 = 1e-4
        self.betaT = 0.02

        # Recommended parameters for specific environments
        self.recommended_params = {
            "temp_sample": {"mdoc": 0.1},
            "Ndiffuse": {"mdoc": 100},
            "Nsample": {"mdoc": 8192},
            "Hsample": {"mdoc": 40}
        }

        # Initialize environment parameters
        self._setup_environment()

    def _setup_environment(self):
        """Initialize the environment and related parameters."""
        # Apply recommended parameters if not disabled
        if not self.disable_recommended_params:
            for param_name in ["temp_sample", "Ndiffuse", "Nsample", "Hsample"]:
                recommended_value = self.recommended_params[param_name].get(
                    self.env_name, getattr(self, param_name)
                )
                setattr(self, param_name, recommended_value)
            print(f"Using recommended parameters: temp_sample={self.temp_sample}")

        self.Nx = self.env.observation_size
        self.Nu = self.env.action_size

        # Reset environment to initial state
        self.state_init = self.env.reset()

    def _setup_diffusion_params(self):
        """Calculate diffusion process parameters."""
        # Create tensors on the specified device
        self.betas = torch.linspace(self.beta0, self.betaT, self.Ndiffuse, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(1 - self.alphas_bar)

        # Conditional distribution parameters
        shifted_alphas_bar = torch.roll(self.alphas_bar, 1)
        shifted_alphas_bar[0] = 1.0  # Handle first element
        Sigmas_cond = ((1 - self.alphas) * (1 - torch.sqrt(shifted_alphas_bar)) / (1 - self.alphas_bar))
        self.sigmas_cond = torch.sqrt(Sigmas_cond)
        self.sigmas_cond[0] = 0.0

        print(f"Initial sigma = {self.sigmas[-1].item():.2e}")

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
        eps_u = torch.randn((self.Nsample, self.Hsample, self.Nu), device=self.device)
        Y0s = eps_u * self.sigmas[i] + Ybar_i
        Y0s = torch.clamp(Y0s, -1.0, 1.0)

        # Evaluate sampled trajectories
        rewss = []
        qs = []
        for j in range(self.Nsample):
            # Convert to numpy for environment compatibility
            actions = Y0s[j].cpu().numpy() if self.device != 'cpu' else Y0s[j].numpy()
            rews, q = self.rollout(self.state_init, actions)
            rewss.append(rews)
            qs.append(q)

        rews = torch.tensor(np.mean(rewss, axis=-1), device=self.device)
        rew_std = rews.std()
        rew_std = torch.where(rew_std < 1e-4, torch.tensor(1.0, device=self.device), rew_std)
        rew_mean = rews.mean()
        logp0 = (rews - rew_mean) / rew_std / self.temp_sample

        # Incorporate demonstration if enabled
        if self.enable_demo:
            xref_logpds = torch.tensor([self.env.eval_xref_logpd(q) for q in qs], device=self.device)
            xref_logpds = xref_logpds - xref_logpds.max()
            logpdemo = (
                    (xref_logpds + self.env.rew_xref - rew_mean) / rew_std / self.temp_sample
            )
            demo_mask = logpdemo > logp0
            logp0 = torch.where(demo_mask, logpdemo, logp0)
            logp0 = (logp0 - logp0.mean()) / logp0.std() / self.temp_sample

        # Update trajectory using weighted average
        weights = torch.nn.functional.softmax(logp0, dim=0)
        Ybar = torch.einsum("n,nij->ij", weights, Y0s)

        # Compute score function and update
        score = 1 / (1.0 - self.alphas_bar[i]) * (-Yi + torch.sqrt(self.alphas_bar[i]) * Ybar)
        Yim1 = 1 / torch.sqrt(self.alphas[i]) * (Yi + (1.0 - self.alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / torch.sqrt(self.alphas_bar[i - 1])

        return Ybar_im1, rews.mean().item()

    def rollout(self, state, actions):
        """Rollout trajectory given initial state and actions."""
        rews = []
        states = [state]
        for t in range(actions.shape[0]):
            state, rew, done, _ = self.env.step(state, actions[t])
            rews.append(rew)
            states.append(state)
            if done:
                break
        return np.array(rews), states

    def optimize(self, render=True):
        """
        Run the full diffusion-based trajectory optimization.
        Args:
            render: Whether to render/save the final trajectory

        Returns:
            Final optimized trajectory and its mean reward
        """
        self._setup_diffusion_params()
        # Initialize with zero trajectory
        YN = torch.zeros((self.Hsample, self.Nu), device=self.device)
        # Run reverse diffusion
        Yi = YN
        Ybars = []
        rewards = []
        with tqdm(range(self.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                Yi, rew = self._reverse_diffusion_step(i, Yi)
                Ybars.append(Yi)
                rewards.append(rew)
                pbar.set_postfix({"rew": f"{rew:.2e}"})

        Ybars = torch.stack(Ybars)
        # Evaluate final reward
        final_actions = Ybars[-1].cpu().numpy() if self.device != 'cpu' else Ybars[-1].numpy()
        rewss_final, _ = self.rollout(self.state_init, final_actions)
        rew_final = np.mean(rewss_final)
        return Ybars[-1], rew_final
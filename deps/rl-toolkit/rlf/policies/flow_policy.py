import torch
from rlf.policies.base_net_policy import BaseNetPolicy
import rlf.rl.utils as rutils

class CreateAction():
    def __init__(self, action):
        self.action = action
        self.hxs = {}
        self.extra = {}
        self.take_action = action
        
    def clip_action(self, low_bound, upp_bound):
        # When CUDA is enabled the action will be on the GPU.
        self.action = rutils.multi_dim_clip(
            self.action,
            low_bound.to(self.action.device),
            upp_bound.to(self.action.device),
        )
        self.take_action = rutils.multi_dim_clip(self.take_action, low_bound, upp_bound)

class FlowPolicy(BaseNetPolicy):
    """
    behavior clone flow policy
    """

    def __init__(self, fuse_states=[], use_goal=False, get_base_net_fn=None):
        super().__init__(use_goal, fuse_states, get_base_net_fn)
        self.state_norm_fn = lambda x: x
        self.action_denorm_fn = lambda x: x

    def set_state_norm_fn(self, state_norm_fn):
        self.state_norm_fn = state_norm_fn

    def set_action_denorm_fn(self, action_denorm_fn):
        self.action_denorm_fn = action_denorm_fn

    def init(self, obs_shape, action_space, args):
        # super().init(obs_shape, action_space, args)
        self.args = args
        self.obs_space, self.action_space = obs_shape, action_space
        self.velocity_net = self.get_base_net_fn
        self.optimizer = torch.optim.Adam(self.velocity_net.parameters(), lr=1e-4)

    def get_add_args(self, parser):
        parser.add_argument(
            "--flow_integration_steps", action="store_true", default=32
        )

    def get_action(self, state, add_state, rnn_hxs, mask, step_info):
        device = self.args.device
        batch_size = state.shape[0]
        action_dim = self.velocity_net.action_dim

        x = torch.randn(batch_size, action_dim, device=device)
        n_steps = getattr(self.args, "flow_integration_steps", 32)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((batch_size, 1), fill_value=i * dt, device=device)
            v = self.forward(x, state, t)  
            x = x + v * dt

        return CreateAction(x)


    def get_storage_hidden_states(self):
        hxs = {}
        hxs['hxs'] = self.velocity_net.hidden_dim
        hxs['txs'] = self.velocity_net.time_dim
        return hxs


    def forward(self, x_t: torch.Tensor, s: torch.Tensor, t: torch.Tensor):
        return self.velocity_net(x_t, s, t)
        

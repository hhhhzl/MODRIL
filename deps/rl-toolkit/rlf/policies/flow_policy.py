import torch
from rlf.policies.base_net_policy import BaseNetPolicy


class FlowPolicy(BaseNetPolicy):
    """
    behavior clone flow policy
    """

    def __init__(self, fuse_states=[], use_goal=False, get_base_net_fn=None):
        super().__init__(use_goal, fuse_states, get_base_net_fn)

    def init(self, obs_shape, action_space, args):
        super().init(obs_shape, action_space, args)
        state_feat_dim = self.base_net.output_shape[0]
        action_dim = action_space.shape[0]
        self.velocity_net = self.get_base_net_fn
        self.velocity_net.state_dim = state_feat_dim
        self.velocity_net.action_dim = action_dim
        self.velocity_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.velocity_net.parameters(), lr=1e-4)

    def get_actions(self, obs, rnn_hxs, masks, **kwargs):
        x_t = kwargs["x_t"]
        t = kwargs["t"]
        state_feat, hxs = self.base_net(obs, rnn_hxs, masks)
        v_pred = self.velocity_net(x_t, state_feat, t)
        return v_pred, hxs, {}

import sys

sys.path.insert(0, "/")
import d4rl
from functools import partial
from rlf.run_settings import RunSettings
from rlf import run_policy
from rlf.args import str2bool
from rlf.policies import RandomPolicy
from rlf.policies.action_replay_policy import ActionReplayPolicy
from rlf.rl.loggers.base_logger import BaseLogger
from rlf.rl.loggers.wb_logger import (
    WbLogger,
    get_wb_ray_config,
    get_wb_ray_kwargs
)
from rlf.algos import (
    BaseILAlgo,
    BaseAlgo,
    NestedAlgo,
    DRAIL_UN,
    DRAIL,
    GAIFO,
    SQIL,
    SAC,
    GAIL,
    PPO,
    BehavioralCloning,
    BehaviorCloneFlowMatching,
    DiffPolicy,
    BehavioralCloningFromObs,
    PWIL,
)
from deps.baselines.ebil.ebil import EBIL
from modril.flowril.flowril import FLOWRIL
from modril.utils.goal_traj_saver import GoalTrajSaver
from modril.utils.trim_trans import trim_episodes_trans
from deps.baselines.get_policy import (
    get_ppo_policy,
    get_basic_policy,
    get_diffusion_policy,
    get_deep_sac_policy,
    get_deep_basic_policy,
    get_bcf_policy
)

# has to import for pick\push resize
import modril.envs.fetch
import modril.envs.goal_check

def get_setup_dict():
    return {
        "pwil": (PWIL(), get_ppo_policy),
        "gaifo": (GAIFO(), get_ppo_policy),
        "ppo": (PPO(), get_ppo_policy),
        "action-replay": (BaseAlgo(), lambda env_name, _: ActionReplayPolicy()),
        "rnd": (BaseAlgo(), lambda env_name, _: RandomPolicy()),
        "dp": (DiffPolicy(), partial(get_diffusion_policy, is_stoch=False)),
        "bco": (BehavioralCloningFromObs(), partial(get_basic_policy, is_stoch=True)),
        "bc-deep": (BehavioralCloning(), get_deep_basic_policy),
        "sqil-deep": (SQIL(), get_deep_sac_policy),
        "sac": (SAC(), get_deep_sac_policy),

        "bc": (BehavioralCloning(), partial(get_basic_policy, is_stoch=False)),
        "bcf": (BehaviorCloneFlowMatching(), partial(get_bcf_policy, time_embed_dim=128)),
        "gail": (GAIL(), get_ppo_policy),
        "drail-un": (DRAIL_UN(), get_ppo_policy),
        "drail": (DRAIL(), get_ppo_policy),
        "ebil": (EBIL(), get_ppo_policy),
        "flowril": (FLOWRIL(), get_ppo_policy),
    }


class BaselinesSettings(RunSettings):
    def get_policy(self):
        return get_setup_dict()[self.base_args.alg][1](
            self.base_args.env_name, self.base_args
        )

    def create_traj_saver(self, save_path):
        return GoalTrajSaver(save_path, False)

    def get_algo(self):
        algo = get_setup_dict()[self.base_args.alg][0]
        if isinstance(algo, NestedAlgo) and isinstance(algo.modules[0], BaseILAlgo):
            algo.modules[0].set_transform_dem_dataset_fn(trim_episodes_trans)
        if isinstance(algo, SQIL):
            algo.il_algo.set_transform_dem_dataset_fn(trim_episodes_trans)
        return algo

    def get_logger(self):
        if self.base_args.no_wb:
            return BaseLogger()
        else:
            return WbLogger(should_log_vids=True)

    def get_add_args(self, parser):
        parser.add_argument("--alg")
        parser.add_argument("--env-name")
        parser.add_argument("--gw-img", type=str2bool, default=True)
        parser.add_argument("--no-wb", action="store_true", default=False)
        parser.add_argument("--freeze-policy", type=str2bool, default=False)
        parser.add_argument("--rollout-agent", type=str2bool, default=False)
        parser.add_argument("--hidden-dim", type=int, default=256)
        parser.add_argument("--depth", type=int, default=2)
        parser.add_argument("--ppo-hidden-dim", type=int, default=64)
        parser.add_argument("--ppo-layers", type=int, default=2)

    def import_add(self):
        import modril.envs.fetch
        import modril.envs.goal_check

    def get_add_ray_config(self, config):
        if self.base_args.no_wb:
            return config
        return get_wb_ray_config(config)

    def get_add_ray_kwargs(self):
        if self.base_args.no_wb:
            return {}
        return get_wb_ray_kwargs()


if __name__ == "__main__":
    run_policy(BaselinesSettings())
from gym.envs.registration import register

register(
    id='Sine-v0',
    entry_point='modril.toy.env:get_sine_env',
    max_episode_steps=1000,
    kwargs={
        'ref_min_score': 0,
        'ref_max_score': 0,
        'dataset_url': 'None'
    }
)

# AntReach
register(
    id='AntGoal-v0',
    entry_point='modril.envs.mujoco.ant:AntGoalEnv',
    max_episode_steps=50,
)

# Maze (Import by d4rl)

# Hopper (hopper v3)
# register(
#     id='HopperCustum-v0',
#     entry_point='modril.envs.mujoco.hopper:HopperEnv',
# )

# Fetch Pick/Push
from modril.envs.fetch import *

# Halfcheetah

# Walker (Import by gym)

# Handrotate
# from modril.envs.hand import *

# Humanoid (hopper v3)
# register(
#     id='HumanoidCustum-v0',
#     entry_point='modril.envs.mujoco.humanoid:HumanoidEnv',
#     max_episode_steps=50,
# )
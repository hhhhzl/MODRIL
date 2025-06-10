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

register(
    id='AntGoal-v0',
    entry_point='modril.envs.ant:AntGoalEnv',
    max_episode_steps=50,
    )
from gym.envs.registration import register
register(
    id='VizFetchPickAndPlaceCustom-v0',
    entry_point='modril.envs.viz.pick:VizFetchPickAndPlaceEnv',
    max_episode_steps=50,
    )

register(
    id='VizFetchPushEnv-v0',
    entry_point='modril.envs.viz.push:VizFetchPushEnv',
    max_episode_steps=60,
    )



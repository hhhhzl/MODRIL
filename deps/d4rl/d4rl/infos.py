"""
This file holds all URLs and reference scores.
"""

#TODO(Justin): This is duplicated. Make all __init__ file URLs and scores point to this file.

DATASET_URLS = {
    'maze2d-open-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-sparse.hdf5',
    'maze2d-umaze-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5',
    'maze2d-medium-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5',
    'maze2d-large-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5',
    'maze2d-eval-umaze-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-umaze-sparse-v1.hdf5',
    'maze2d-eval-medium-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-medium-sparse-v1.hdf5',
    'maze2d-eval-large-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-large-sparse-v1.hdf5',
    'maze2d-open-dense-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-dense.hdf5',
    'maze2d-umaze-dense-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-dense-v1.hdf5',
    'maze2d-medium-dense-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-dense-v1.hdf5',
    'maze2d-large-dense-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-dense-v1.hdf5',
    'maze2d-eval-umaze-dense-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-umaze-dense-v1.hdf5',
    'maze2d-eval-medium-dense-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-medium-dense-v1.hdf5',
    'maze2d-eval-large-dense-v1' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-eval-large-dense-v1.hdf5',
    'minigrid-fourrooms-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/minigrid/minigrid4rooms.hdf5',
    'minigrid-fourrooms-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/minigrid/minigrid4rooms_random.hdf5',
    'pen-human-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-v0_demos_clipped.hdf5',
    'pen-cloned-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-demos-v0-bc-combined.hdf5',
    'pen-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/pen-v0_expert_clipped.hdf5',
    'hammer-human-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-v0_demos_clipped.hdf5',
    'hammer-cloned-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-demos-v0-bc-combined.hdf5',
    'hammer-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/hammer-v0_expert_clipped.hdf5',
    'relocate-human-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-v0_demos_clipped.hdf5',
    'relocate-cloned-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-demos-v0-bc-combined.hdf5',
    'relocate-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/relocate-v0_expert_clipped.hdf5',
    'door-human-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-v0_demos_clipped.hdf5',
    'door-cloned-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-demos-v0-bc-combined.hdf5',
    'door-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg/door-v0_expert_clipped.hdf5',
    'halfcheetah-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_random.hdf5',
    'halfcheetah-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium.hdf5',
    'halfcheetah-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_expert.hdf5',
    'halfcheetah-medium-replay-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_mixed.hdf5',
    'halfcheetah-medium-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium_expert.hdf5',
    'walker2d-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_random.hdf5',
    'walker2d-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium.hdf5',
    'walker2d-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_expert.hdf5',
    'walker2d-medium-replay-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker_mixed.hdf5',
    'walker2d-medium-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium_expert.hdf5',
    'hopper-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_random.hdf5',
    'hopper-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium.hdf5',
    'hopper-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_expert.hdf5',
    'hopper-medium-replay-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_mixed.hdf5',
    'hopper-medium-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium_expert.hdf5',
    'antReach-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_random.hdf5',
    'antReach-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_medium.hdf5',
    'antReach-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_expert.hdf5',
    'antReach-medium-replay-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_mixed.hdf5',
    'antReach-medium-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_medium_expert.hdf5',
    'antReach-random-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_random_expert.hdf5',
    'antmaze-umaze-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
    'antmaze-umaze-diverse-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
    'antmaze-medium-play-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
    'antmaze-medium-diverse-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
    'antmaze-large-play-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5',
    'antmaze-large-diverse-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse.hdf5',
    'flow-ring-random-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-ring-v0-random.hdf5',
    'flow-ring-controller-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-ring-v0-idm.hdf5',
    'flow-merge-random-v0':'http://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-merge-v0-random.hdf5',
    'flow-merge-controller-v0':'http://rail.eecs.berkeley.edu/datasets/offline_rl/flow/flow-merge-v0-idm.hdf5',
    'kitchen-complete-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5',
    'kitchen-partial-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_light_slider-v0.hdf5',
    'kitchen-mixed-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5',
    'carla-lane-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_lane_follow_flat-v0.hdf5',
    'carla-town-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_town_subsamp_flat-v0.hdf5',
    'carla-town-full-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/carla/carla_town_flat-v0.hdf5',
    'bullet-halfcheetah-random-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_random.hdf5',
    'bullet-halfcheetah-medium-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_medium.hdf5',
    'bullet-halfcheetah-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_expert.hdf5',
    'bullet-halfcheetah-medium-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_medium_expert.hdf5',
    'bullet-halfcheetah-medium-replay-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-halfcheetah_medium_replay.hdf5',
    'bullet-hopper-random-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_random.hdf5',
    'bullet-hopper-medium-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_medium.hdf5',
    'bullet-hopper-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_expert.hdf5',
    'bullet-hopper-medium-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_medium_expert.hdf5',
    'bullet-hopper-medium-replay-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-hopper_medium_replay.hdf5',
    'bullet-antReach-random-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_random.hdf5',
    'bullet-antReach-medium-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_medium.hdf5',
    'bullet-antReach-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_expert.hdf5',
    'bullet-antReach-medium-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_medium_expert.hdf5',
    'bullet-antReach-medium-replay-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-ant_medium_replay.hdf5',
    'bullet-walker2d-random-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_random.hdf5',
    'bullet-walker2d-medium-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_medium.hdf5',
    'bullet-walker2d-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_expert.hdf5',
    'bullet-walker2d-medium-expert-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_medium_expert.hdf5',
    'bullet-walker2d-medium-replay-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-walker2d_medium_replay.hdf5',
    'bullet-maze2d-open-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-maze2d-open-sparse.hdf5',
    'bullet-maze2d-umaze-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-maze2d-umaze-sparse.hdf5',
    'bullet-maze2d-medium-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-maze2d-medium-sparse.hdf5',
    'bullet-maze2d-large-v0': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/bullet/bullet-maze2d-large-sparse.hdf5',
}


REF_MIN_SCORE = {
    'maze2d-open-v0' : 0.01 ,
    'maze2d-umaze-v1' : 23.85 ,
    'maze2d-medium-v1' : 13.13 ,
    'maze2d-large-v1' : 6.7 ,
    'maze2d-open-dense-v0' : 11.17817 ,
    'maze2d-umaze-dense-v1' : 68.537689 ,
    'maze2d-medium-dense-v1' : 44.264742 ,
    'maze2d-large-dense-v1' : 30.569041 ,
    'minigrid-fourrooms-v0' : 0.01442 ,
    'minigrid-fourrooms-random-v0' : 0.01442 ,
    'pen-human-v0' : 96.262799 ,
    'pen-cloned-v0' : 96.262799 ,
    'pen-expert-v0' : 96.262799 ,
    'hammer-human-v0' : -274.856578 ,
    'hammer-cloned-v0' : -274.856578 ,
    'hammer-expert-v0' : -274.856578 ,
    'relocate-human-v0' : -6.425911 ,
    'relocate-cloned-v0' : -6.425911 ,
    'relocate-expert-v0' : -6.425911 ,
    'door-human-v0' : -56.512833 ,
    'door-cloned-v0' : -56.512833 ,
    'door-expert-v0' : -56.512833 ,
    'halfcheetah-random-v0' : -280.178953 ,
    'halfcheetah-medium-v0' : -280.178953 ,
    'halfcheetah-expert-v0' : -280.178953 ,
    'halfcheetah-medium-replay-v0' : -280.178953 ,
    'halfcheetah-medium-expert-v0' : -280.178953 ,
    'walker2d-random-v0' : 1.629008 ,
    'walker2d-medium-v0' : 1.629008 ,
    'walker2d-expert-v0' : 1.629008 ,
    'walker2d-medium-replay-v0' : 1.629008 ,
    'walker2d-medium-expert-v0' : 1.629008 ,
    'hopper-random-v0' : -20.272305 ,
    'hopper-medium-v0' : -20.272305 ,
    'hopper-expert-v0' : -20.272305 ,
    'hopper-medium-replay-v0' : -20.272305 ,
    'hopper-medium-expert-v0' : -20.272305 ,
    'antReach-random-v0' : -325.6,
    'antReach-medium-v0' : -325.6,
    'antReach-expert-v0' : -325.6,
    'antReach-medium-replay-v0' : -325.6,
    'antReach-medium-expert-v0' : -325.6,
    'antmaze-umaze-v0' : 0.0 ,
    'antmaze-umaze-diverse-v0' : 0.0 ,
    'antmaze-medium-play-v0' : 0.0 ,
    'antmaze-medium-diverse-v0' : 0.0 ,
    'antmaze-large-play-v0' : 0.0 ,
    'antmaze-large-diverse-v0' : 0.0 ,
    'kitchen-complete-v0' : 0.0 ,
    'kitchen-partial-v0' : 0.0 ,
    'kitchen-mixed-v0' : 0.0 ,
    'flow-ring-random-v0' : -165.22 ,
    'flow-ring-controller-v0' : -165.22 ,
    'flow-merge-random-v0' : 118.67993 ,
    'flow-merge-controller-v0' : 118.67993 ,
    'carla-lane-v0': -0.8503839912088142,
    'carla-town-v0': -114.81579500772153, # random score
    'bullet-halfcheetah-random-v0': -1275.766996,
    'bullet-halfcheetah-medium-v0': -1275.766996,
    'bullet-halfcheetah-expert-v0': -1275.766996,
    'bullet-halfcheetah-medium-expert-v0': -1275.766996,
    'bullet-halfcheetah-medium-replay-v0': -1275.766996,
    'bullet-hopper-random-v0': 20.058972,
    'bullet-hopper-medium-v0': 20.058972,
    'bullet-hopper-expert-v0': 20.058972,
    'bullet-hopper-medium-expert-v0': 20.058972,
    'bullet-hopper-medium-replay-v0': 20.058972,
    'bullet-antReach-random-v0': 373.705955,
    'bullet-antReach-medium-v0': 373.705955,
    'bullet-antReach-expert-v0': 373.705955,
    'bullet-antReach-medium-expert-v0': 373.705955,
    'bullet-antReach-medium-replay-v0': 373.705955,
    'bullet-walker2d-random-v0': 16.523877,
    'bullet-walker2d-medium-v0': 16.523877,
    'bullet-walker2d-expert-v0': 16.523877,
    'bullet-walker2d-medium-expert-v0': 16.523877,
    'bullet-walker2d-medium-replay-v0': 16.523877,
    'bullet-maze2d-open-v0': 8.750000,
    'bullet-maze2d-umaze-v0': 32.460000,
    'bullet-maze2d-medium-v0': 14.870000,
    'bullet-maze2d-large-v0': 1.820000,
}

REF_MAX_SCORE = {
    'maze2d-open-v0' : 20.66 ,
    'maze2d-umaze-v1' : 161.86 ,
    'maze2d-medium-v1' : 277.39 ,
    'maze2d-large-v1' : 273.99 ,
    'maze2d-open-dense-v0' : 27.166538620695782 ,
    'maze2d-umaze-dense-v1' : 193.66285642381482 ,
    'maze2d-medium-dense-v1' : 297.4552547777125 ,
    'maze2d-large-dense-v1' : 303.4857382709002 ,
    'minigrid-fourrooms-v0' : 2.89685 ,
    'minigrid-fourrooms-random-v0' : 2.89685 ,
    'pen-human-v0' : 3076.8331017826877 ,
    'pen-cloned-v0' : 3076.8331017826877 ,
    'pen-expert-v0' : 3076.8331017826877 ,
    'hammer-human-v0' : 12794.134825156867 ,
    'hammer-cloned-v0' : 12794.134825156867 ,
    'hammer-expert-v0' : 12794.134825156867 ,
    'relocate-human-v0' : 4233.877797728884 ,
    'relocate-cloned-v0' : 4233.877797728884 ,
    'relocate-expert-v0' : 4233.877797728884 ,
    'door-human-v0' : 2880.5693087298737 ,
    'door-cloned-v0' : 2880.5693087298737 ,
    'door-expert-v0' : 2880.5693087298737 ,
    'halfcheetah-random-v0' : 12135.0 ,
    'halfcheetah-medium-v0' : 12135.0 ,
    'halfcheetah-expert-v0' : 12135.0 ,
    'halfcheetah-medium-replay-v0' : 12135.0 ,
    'halfcheetah-medium-expert-v0' : 12135.0 ,
    'walker2d-random-v0' : 4592.3 ,
    'walker2d-medium-v0' : 4592.3 ,
    'walker2d-expert-v0' : 4592.3 ,
    'walker2d-medium-replay-v0' : 4592.3 ,
    'walker2d-medium-expert-v0' : 4592.3 ,
    'hopper-random-v0' : 3234.3 ,
    'hopper-medium-v0' : 3234.3 ,
    'hopper-expert-v0' : 3234.3 ,
    'hopper-medium-replay-v0' : 3234.3 ,
    'hopper-medium-expert-v0' : 3234.3 ,
    'antReach-random-v0' : 3879.7,
    'antReach-medium-v0' : 3879.7,
    'antReach-expert-v0' : 3879.7,
    'antReach-medium-replay-v0' : 3879.7,
    'antReach-medium-expert-v0' : 3879.7,
    'antmaze-umaze-v0' : 1.0 ,
    'antmaze-umaze-diverse-v0' : 1.0 ,
    'antmaze-medium-play-v0' : 1.0 ,
    'antmaze-medium-diverse-v0' : 1.0 ,
    'antmaze-large-play-v0' : 1.0 ,
    'antmaze-large-diverse-v0' : 1.0 ,
    'kitchen-complete-v0' : 4.0 ,
    'kitchen-partial-v0' : 4.0 ,
    'kitchen-mixed-v0' : 4.0 ,
    'flow-ring-random-v0' : 24.42 ,
    'flow-ring-controller-v0' : 24.42 ,
    'flow-merge-random-v0' : 330.03179 ,
    'flow-merge-controller-v0' : 330.03179 ,
    'carla-lane-v0': 1023.5784385429523,
    'carla-town-v0': 2440.1772022247314, # avg dataset score
    'bullet-halfcheetah-random-v0': 2381.6725,
    'bullet-halfcheetah-medium-v0': 2381.6725,
    'bullet-halfcheetah-expert-v0': 2381.6725,
    'bullet-halfcheetah-medium-expert-v0': 2381.6725,
    'bullet-halfcheetah-medium-replay-v0': 2381.6725,
    'bullet-hopper-random-v0': 1441.8059623430963,
    'bullet-hopper-medium-v0': 1441.8059623430963,
    'bullet-hopper-expert-v0': 1441.8059623430963,
    'bullet-hopper-medium-expert-v0': 1441.8059623430963,
    'bullet-hopper-medium-replay-v0': 1441.8059623430963,
    'bullet-antReach-random-v0': 2650.495,
    'bullet-antReach-medium-v0': 2650.495,
    'bullet-antReach-expert-v0': 2650.495,
    'bullet-antReach-medium-expert-v0': 2650.495,
    'bullet-antReach-medium-replay-v0': 2650.495,
    'bullet-walker2d-random-v0': 1623.6476303317536,
    'bullet-walker2d-medium-v0': 1623.6476303317536,
    'bullet-walker2d-expert-v0': 1623.6476303317536,
    'bullet-walker2d-medium-expert-v0': 1623.6476303317536,
    'bullet-walker2d-medium-replay-v0': 1623.6476303317536,
    'bullet-maze2d-open-v0': 64.15,
    'bullet-maze2d-umaze-v0': 153.99,
    'bullet-maze2d-medium-v0': 238.05,
    'bullet-maze2d-large-v0': 285.92,
}


#Gym-MuJoCo V1 envs
for env in ['halfcheetah', 'hopper', 'walker2d', 'antReach']:
    for dset in ['random', 'medium', 'expert', 'medium-replay', 'full-replay', 'medium-expert']:
        dset_name = env+'_'+dset.replace('-', '_')+'-v1'
        env_name = dset_name.replace('_', '-')
        DATASET_URLS[env_name] = 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v1/%s.hdf5' % dset_name
        REF_MIN_SCORE[env_name] = REF_MIN_SCORE[env+'-random-v0']
        REF_MAX_SCORE[env_name] = REF_MAX_SCORE[env+'-random-v0']

#Adroit v1 envs
for env in ['hammer', 'pen', 'relocate', 'door']:
    for dset in ['human', 'expert', 'cloned']:
        env_name = env+'-'+dset+'-v1'
        DATASET_URLS[env_name] = 'http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/%s.hdf5' % env_name
        REF_MIN_SCORE[env_name] = REF_MIN_SCORE[env+'-human-v0']
        REF_MAX_SCORE[env_name] = REF_MAX_SCORE[env+'-human-v0']


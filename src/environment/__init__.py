from gymnasium.envs.registration import register

register(
    id='VideoBasedPothole-v0',
    entry_point='src.environment.pothole_env:VideoBasedPotholeEnv',
    max_episode_steps=1,
    kwargs={'split': 'train'}
)

register(
    id='VideoBasedPothole-val-v0',
    entry_point='src.environment.pothole_env:VideoBasedPotholeEnv',
    max_episode_steps=1,
    kwargs={'split': 'val'}
)

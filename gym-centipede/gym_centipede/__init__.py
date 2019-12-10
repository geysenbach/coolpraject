from gym.envs.registration import register

register(
    id='centipede-v0',
    entry_point='gym_centipede.envs:CentipedeEnv',
)

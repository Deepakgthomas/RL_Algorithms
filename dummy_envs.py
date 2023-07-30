import os
import sys
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

if __name__ == '__main__':

    def make_env():
        return gym.make('CartPole-v1')

    if __name__ == '__main__':
        num_envs = 4  # Number of parallel environments
        env_fns = [make_env for _ in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns)
        obs = vec_env.reset()
        done = [False] * num_envs
        while not all(done):
            actions = [vec_env.action_space.sample() for _ in range(num_envs)]
            next_obs, rewards, done, info = vec_env.step(actions)
            print(done, info)
            obs = next_obs


# obs = env.reset_infos()
# print("obs = ", obs)
# done = False
# while not done:
#     action = env.action_space.sample()
#     print("action = ", action)
#     next_obs, reward, done, info = env.step(action)
#     obs = next_obs
# # observation, reward, terminated, truncated = env.step([1,2])
# # print("observation  = ", observation)

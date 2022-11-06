# Taken from - https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html

import gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make("Pendulum-v1")

# The noise objects for DDPG
print("env.action_space.shape[-1] = ", env.action_space.shape[-1])
print("env.action_space = ", env.action_space)

action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
# model.learn(total_timesteps=10000, log_interval=10)
# model.save("ddpg_pendulum")
# env = model.get_env()
#
# del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    print("action = ", action)
    obs, rewards, dones, info = env.step(action)
    print("rewards = ", rewards)
    # env.render()
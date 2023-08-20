import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
import envpool
print(envpool.__version__)
print(envpool.list_all_envs())
vec_env = envpool.make("Pong-v5", env_type = "gym", num_envs = 100)

# # Parallel environments
# vec_env = gym.make("PongNoFrameskip-v4")
# # vec_env = make_vec_env("PongNoFrameskip-v4", n_envs=2, seed=3)
# vec_env = AtariWrapper(vec_env)
model = PPO("CnnPolicy", vec_env, verbose=1, n_steps=128, n_epochs=4,
            batch_size=256, learning_rate=2.5e-4, clip_range=0.1,
            vf_coef=0.5, ent_coef=0.01)
model.learn(total_timesteps=1e7)
model.save("ppo_cartpole")

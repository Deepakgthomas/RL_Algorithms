#Modified this code - https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Chapter%204/Ch4_book.ipynb
#Also, modified this code - https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
# Also, modified this code - https://github.com/ericyangyu/PPO-for-Beginners/blob/9abd435771aa84764d8d0d1f737fa39118b74019/ppo.py#L151
# Got a lot of help from the subreddit - reinforcement_learning


if __name__ == '__main__':

    import numpy as np
    import gymnasium as gym
    import imageio

    from gymnasium.wrappers import AtariPreprocessing
    import torch
    import random
    import matplotlib.pyplot as plt
    from torch import nn
    import torchvision as tv
    import torch.nn.functional as F
    torch.manual_seed(798)
    import matplotlib.pyplot as plt
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    from collections import deque
    num_envs = 1
    ent_coeff = 0.01
    num_channels = 1
    batches = 4
    channels = 3
    learning_rate = 0.00025
    episodes = 100
    gae_lambda = 0.95
    gamma = 0.99
    clip = 0.2
    rollout_steps = 100
    training_iters = 4
    actor_PATH = './actor_model' + 'breakout' + '.pt'
    critic_PATH = 'critic_model.pt'

    device = torch.device("cpu")

    env = gym.make("BreakoutNoFrameskip-v4", render_mode = "rgb_array")
    gif_path = './saved_rl_video' + 'breakout' + '.gif'

    env = AtariPreprocessing(env)
    square_size = env.observation_space.shape[-1]

    class Actor(nn.Module):
        def __init__(self, state_size, action_size):
            super(Actor, self).__init__()
            self.state_size = state_size
            self.action_size = action_size
            self.conv1 = nn.Conv2d(num_channels, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.conv3 = nn.Conv2d(16, 32, 3)
            self.fc1 = nn.Linear(2048, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, action_size)
            self.last = nn.Softmax(dim=-1)

        def forward(self,x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            x = self.last(x)
            return x


    actor = Actor(env.observation_space.shape[-1], env.action_space.n).to(device)
    actor.load_state_dict(torch.load(actor_PATH))
    actor.eval()

    obs = torch.tensor(env.reset()[0], dtype=torch.float32).to(device)

    tot_rewards = np.array([0 for i in range(num_envs)], dtype=float)
    final_scores = []
    last_n_rewards = deque(maxlen=10)
    obs = obs.reshape(num_envs, 1, square_size, square_size)
    frames = []
    for _ in range(1000):

        act_probs = torch.distributions.Categorical(actor(obs).squeeze())
        action = act_probs.sample().squeeze()
        action = action.cpu().detach().numpy()
        print("Action = ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        obs = obs.reshape(num_envs, 1, square_size, square_size)
        obs = torch.tensor(obs, dtype=torch.float32)
        tot_rewards += reward
        frame = env.render()
        frames.append(frame)

    if terminated or truncated:
        obs, info = env.reset()
        final_scores.append(tot_rewards)
        tot_rewards = np.array([0 for i in range(num_envs)], dtype=float)

    env.close()
    imageio.mimsave(gif_path, frames, duration=0.1)



















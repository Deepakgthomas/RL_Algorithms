#Modified this code - https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Chapter%204/Ch4_book.ipynb
#Also, modified this code - https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
# Also, modified this code - https://github.com/ericyangyu/PPO-for-Beginners/blob/9abd435771aa84764d8d0d1f737fa39118b74019/ppo.py#L151
import numpy as np
import gym
import torch
from torch import nn
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 0.0001
episodes = 10000

def discount_rewards(reward, gamma = 0.99):
    return torch.pow(gamma, torch.arange(len(reward)))*reward
# def normalize_rewards(disc_reward):
#     return disc_reward/(disc_reward.max())

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size, 300),
            nn.ReLU(),
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax()
        )
    def forward(self,x):
        x = self.linear_relu_stack(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_stack = nn.Sequential(
            nn.Linear(state_size, 300),
            nn.ReLU(),
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.linear_stack(x)
        return x


def rollout():
    batch_obs = []
    batch_act = []
    batch_log_probs = []
    batch_rews = []
    batch_rtgs = []
    batch_lens = []
    ep_rews = []
    obs = env.reset()
    while True:
        batch_obs.append(obs)
        act_probs = torch.distributions.Categorical(actor(obs))
        action = act_probs.sample()
        next_state, reward, done, info = env.step(action)
        ep_rews.append(reward)
        batch_act.append(action)
        batch_log_probs.append(act_probs.log_prob(action))
        if done:
            break
    batch_rews.append(ep_rews)
    for i in batch_rews:
        disc_reward = discount_rewards(i)
        batch_rtgs.append(disc_reward)

    return batch_obs, batch_rews, batch_act, batch_log_probs, batch_rtgs

actor = Actor(env.observation_space.shape[0], env.action_space.n).to(device)
critic = Critic(env.observation_space.shape[0], env.action_space.n).to(device)
policy_opt = torch.optim.Adam(params = actor.parameters(), lr = learning_rate)
value_opt = torch.optim.Adam(params = critic.parameters(), lr = learning_rate)

score = []
for i in range(episodes):
    batch_obs, batch_rews, batch_act, batch_log_probs, batch_rtgs = rollout()
    value = critic(batch_obs)
    # Why are we detaching value
    A_k = batch_rtgs - value.detach()

    for _ in range(10):
        value = critic(batch_obs)
        act_probs = torch.distributions.Categorical(actor(batch_obs))

        #todo - Just copied this code. Not sure what's going on over here -
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        ratios = torch.exp(log_probs -batch_log_probs)
        surr1 = ratios*A_k
        surr2 = torch.clamp(ratios, 1 - clip, 1 + clip)*A_k
        


        pass

















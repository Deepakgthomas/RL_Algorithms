# Following the algorithm from here - https://spinningup.openai.com/en/latest/algorithms/sac.html
#This is supposed to be a bare bones implementation. So it doesn't have -

#1. Target Networks
#2. Clipped Q Networks
#3. Multiple Q Networks
# Here we import all libraries
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import torch
import random
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import deque
from torch.distributions.normal import Normal
import torchvision as tv
import torch.nn.functional as F
import torch.optim as optim
import sys
value_lr = 0.001
policy_lr = 0.001
batch_size = 100
episodes = 1000
ent_coeff = 0.2 #taken from cleanrl
gamma = 0.99
Q_learning_rate = 0.001
replay_buffer = deque(maxlen=10000000)
tot_rewards = []


env = gym.make("Pendulum-v1")
# print("env.observation_space.shape = ", env.observation_space.shape[0])
# print("env = ", env.action_space.shape[0])
class Q_function(nn.Module):
    def __init__(self, state_size, action_size):
        super(Q_function, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size+action_size, 300),
            nn.ReLU(),
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state, action):
        x = torch.cat((state, action),1)
        x = self.linear_relu_stack(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, dim_state, dim_action, init_w=3e-3):
        super(PolicyNetwork,self).__init__()
        self.linear1 = nn.Linear(dim_state, 32)
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128, dim_action)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, deterministic = False):
        mu = F.relu(self.linear1(state))
        mu = F.relu(self.linear2(mu))
        mu = F.tanh(self.linear3(mu))

        sigma = F.relu(self.linear1(state))
        sigma = F.relu(self.linear2(sigma))
        sigma = F.relu(self.linear3(sigma))
        # print("mu = ", mu)
        # print("sigma = ", sigma)
        dist = Normal(mu, torch.clamp(sigma, min=0.00001))
        if not deterministic:
            action = dist.rsample()
        else:
            action = mu

        #Copying this from here - https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
        #Need to understand it better
        log_pi = dist.log_prob(action).sum(axis=-1)
        log_pi -= (2*(np.log(2)-action-F.softplus(-2*action))).sum(axis=1)

        #todo Confused with the location of log prob

        action = F.tanh(action)




        return action, log_pi

Q1 = Q_function(env.observation_space.shape[-1], 1)

# Q2 = Q_function(env.observation_space.shape[-1], env.action_space.n)

policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0])
Q1_opt = torch.optim.Adam(params = Q1.parameters(), lr = Q_learning_rate)
policy_opt = torch.optim.Adam(params = policy.parameters(), lr = policy_lr)


def update():
    with torch.no_grad():
        state, next_state, reward, done, action = zip(*random.sample(replay_buffer, batch_size))
        state = torch.stack(list(state), dim=0).squeeze(1).reshape(batch_size, -1)
        next_state = torch.from_numpy(np.array(next_state)).reshape(batch_size, -1).type(torch.float32)
        reward = torch.from_numpy(np.array(reward))
        action = torch.from_numpy(np.array(action)).reshape(-1, 1)
        done = torch.from_numpy(np.array(done)).long()

    # a'^{~}
    curr_policy_next_action = policy(next_state)[0]
    # a^{~}
    curr_policy_action = policy(state)[0]
    # Q(s,a)
    current_Q = Q1(state, action).squeeze()
    # Q(s, a^{~})
    current_Q_new = Q1(state, curr_policy_action).squeeze()
    # Q(s, a'^{~})
    next_state_Q = Q1(next_state, curr_policy_next_action).squeeze()
    # y(r, s', d)
    target = reward + gamma*(1-done)*(next_state_Q - ent_coeff*policy(next_state)[1])

    # Q_loss = ((current_Q - target)**2).mean()
    # Q1_opt.zero_grad()
    # Q_loss.backward()
    # Q1_opt.step()
    # policy_loss = (current_Q_new-ent_coeff*torch.log(policy(next_state)[1])).mean()
    # policy_opt.zero_grad()
    # policy_loss.backward()
    # policy_opt.step()



    # Simulataenously summing both Q and policy loss. Otherwise, I was getting an error
    total_loss = ((current_Q - target)**2).mean() + (current_Q_new-ent_coeff*torch.log(policy(state)[1])).mean()
    Q1_opt.zero_grad()
    policy_opt.zero_grad()
    total_loss.backward()
    Q1_opt.step()
    policy_opt.step()


for i in range(episodes):
    print("i = ", i)
    state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
    eps_rew = 0
    done = False
    while not done:

        action = policy(state)[0].detach().numpy()
        next_state, reward, done, _ = env.step(action)
        print("reward = ", reward)
        replay_buffer.append((state, next_state, reward, done, action))
        eps_rew += reward
        if done:
            tot_rewards.append(eps_rew)
            break
        if len(replay_buffer)>batch_size:
            update()
        state = torch.tensor(next_state, dtype=torch.float32).T
    print("Episode reward = ", eps_rew)
    tot_rewards.append(eps_rew)




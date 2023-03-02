# Following the algorithm from here - https://spinningup.openai.com/en/latest/algorithms/sac.html
#Took ideas from -
#1. https://github.com/higgsfield/RL-Adventure-2/blob/master/7.soft%20actor-critic.ipynb
#2. https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
#3. https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
#This is supposed to be a bare bones implementation. So it doesn't have -

#1. Target Networks
#2. Clipped Q Networks
#3. Multiple Q Networks
# Here we import all libraries

#todo How do I deal with starting states? Spinning Up spoke about applying entropy to starting states.
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
value_lr = 2.5e-4
policy_lr = 2.5e-4
batch_size = 500
episodes = 1000
ent_coeff = 0.5 #taken from cleanrl
gamma = 0.99
Q_learning_rate = 2.5e-4
replay_buffer = deque(maxlen=10000000)
mem_size = 8000
tot_rewards = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("Pendulum-v1")
act_limit = env.action_space.high[0]
# print("act_limit = ", act_limit)
# print("env.observation_space.shape = ", env.observation_space.shape[0])
# print("env = ", env.action_space.shape[0])
class Q_function(nn.Module):
    def __init__(self, state_size, action_size, init_w = 3e-3):
        super(Q_function, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size+action_size, 300),
            nn.ReLU(),
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.last_linear = nn.Linear(128, 1)
        self.last_linear.weight.data.uniform_(-init_w, init_w)
        self.last_linear.bias.data.uniform_(-init_w, init_w)
    def forward(self, state, action):
        x = torch.cat((state, action),1)
        x = self.linear_relu_stack(x)
        x = self.last_linear(x)
        return x
LOG_STD_MAX = 2
LOG_STD_MIN = -5
class PolicyNetwork(nn.Module):
    def __init__(self, dim_state, dim_action, act_limit, init_w=3e-3):
        super(PolicyNetwork,self).__init__()
        self.linear1 = nn.Linear(dim_state, 32)
        self.linear2 = nn.Linear(32, 32)
        self.mean = nn.Linear(32,1)
        self.std = nn.Linear(32,1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean(x)
        log_std = self.std(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats


        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = act_limit
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(act_limit * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * act_limit
        return mean, log_prob

Q1 = Q_function(env.observation_space.shape[-1], 1).to(device)

# Q2 = Q_function(env.observation_space.shape[-1], env.action_space.n)

policy = PolicyNetwork(env.observation_space.shape[0], 2, act_limit).to(device)
Q1_opt = torch.optim.Adam(params = Q1.parameters(), lr = Q_learning_rate)
policy_opt = torch.optim.Adam(params = policy.parameters(), lr = policy_lr)


def update():
    with torch.no_grad():
        state, next_state, reward, done, action = zip(*random.sample(replay_buffer, batch_size))
        state = torch.stack(list(state), dim=0).squeeze(1).reshape(batch_size, -1).to(device)
        # print("state shape = ", state.shape)
        next_state = torch.from_numpy(np.array(next_state)).reshape(batch_size, -1).type(torch.float32).to(device)
        # print("next_state shape = ", next_state.shape)

        reward = torch.from_numpy(np.array(reward)).to(device)
        # print("reward shape = ", reward.shape)


        action = torch.from_numpy(np.array(action)).reshape(-1,1).to(device)
        # print("action shape = ", action.shape)


        done = torch.from_numpy(np.array(done)).long().to(device)
        # print("done shape = ", done.shape)


    # a'^{~}
    curr_policy_next_action = policy(next_state)[0]
    # print("curr_policy_next_action = ", curr_policy_next_action.shape)
    # a^{~}
    curr_policy_action = policy(state)[0]
    # print("curr_policy_action = ", curr_policy_action.shape)

    # Q(s,a)
    current_Q = Q1(state, action).squeeze()
    # print("current_Q = ", current_Q.shape)

    # Q(s, a^{~})
    current_Q_new = Q1(state, curr_policy_action).squeeze()
    # print("current_Q_new = ", current_Q_new.shape)

    # Q(s, a'^{~})
    next_state_Q = Q1(next_state, curr_policy_next_action).squeeze()
    # print("next_state_Q = ", next_state_Q.shape)
    log_probs_next_action = policy(next_state)[1]

    log_probs_current_action = policy(state)[1]
    # y(r, s', d)
    target = reward + gamma*(1-done)*(next_state_Q - ent_coeff*log_probs_next_action)

    # Q_loss = ((current_Q - target)**2).mean()
    # Q1_opt.zero_grad()
    # Q_loss.backward()
    # Q1_opt.step()
    # policy_loss = (current_Q_new-ent_coeff*torch.log(policy(next_state)[1])).mean()
    # policy_opt.zero_grad()
    # policy_loss.backward()
    # policy_opt.step()



    # Simulataenously summing both Q and policy loss. Otherwise, I was getting an error
    total_loss = ((current_Q - target)**2).mean() + (current_Q_new-ent_coeff*log_probs_current_action).mean()
    Q1_opt.zero_grad()
    policy_opt.zero_grad()
    total_loss.backward()
    Q1_opt.step()
    policy_opt.step()
check_learning_start = True
for i in range(episodes):
    print("i = ", i)
    state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)

    eps_rew = 0
    done = False
    while not done:

        action = policy(state.to(device))[0].cpu().detach().numpy().reshape(-1)
        next_state, reward, done, _ = env.step(action)
        # print("reward = ", reward)
        replay_buffer.append((state, next_state, reward, done, action))
        eps_rew += reward
        if done:
            tot_rewards.append(eps_rew)
            break
        if len(replay_buffer)>mem_size and check_learning_start:
            print("The learning process has started")
            check_learning_start = False
        if len(replay_buffer)>mem_size:
            update()
        state = torch.tensor(next_state, dtype=torch.float32).squeeze().unsqueeze(0)
    print("Episode reward = ", eps_rew)
    tot_rewards.append(eps_rew)

    if(i%10==0 and i>0):
        plt.scatter(np.arange(len(tot_rewards)), tot_rewards)
        plt.show(block=False)
        plt.pause(3)
        plt.close()




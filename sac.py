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
import torchvision as tv
import torch.nn.functional as F
import torch.optim as optim
import sys
value_lr = 0.001
policy_lr = 0.001
episodes = 1000
env = gym.make("Pendulum-v1")
class Q_function(nn.Module):
    def __init__(self, state_size, action_size):
        super(Q_function, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size, 300),
            nn.ReLU(),
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x):
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

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x

Q1 = Q_function(env.observation_space.shape[-1], env.action_space.n)
# Q2 = Q_function(env.observation_space.shape[-1], env.action_space.n)

policy = PolicyNetwork(env.observation_space.shape[-1], env.action_space.n)

value_opt = optim.Adam(Q1.parameters(), lr=value_lr)
policy_opt = optim.Adam(policy.parameters(), lr = policy_lr)

for i in range(episodes):
    state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
    done = False
    while not done:
        action = policy(state).cpu().detach().numpy()


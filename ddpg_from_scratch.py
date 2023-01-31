# Modifying the code given over here - https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb

import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import random
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
value_lr = 1e-3
policy_lr = 1e-4
mem_size = 1000000
replay_buffer = deque(maxlen=mem_size)
episodes = 50000
batch_size = 500
gamma = 0.99
polyak = 0.995
noise_scale = 10


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class ValueNetwork(nn.Module):
    def __init__(self, dim_state, dim_action, init_w=3e-3):
        super(ValueNetwork,self).__init__()
        self.linear1 = nn.Linear(dim_state+dim_action, 32)
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128,1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
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


def ddpg_update(batch_size):
    state, next_state, reward, done, action = zip(*random.sample(replay_buffer, batch_size))
    state = torch.stack(list(state), dim=0).squeeze(1).to(device)
    # state= state.reshape(batch_size, 3, 210, 160).to(device)
    # next_state = torch.from_numpy(np.array(next_state)).reshape(batch_size, 3, 210, 160).type(torch.float32).to(device)
    next_state = torch.from_numpy(np.array(next_state)).type(torch.float32).to(device)

    reward = torch.from_numpy(np.array(reward)).to(device)

    done = torch.from_numpy(np.array(done)).long().to(device)
    action = torch.from_numpy(np.array(action)).type(torch.int64).to(device)

    policy_loss = -online_value(state, online_policy(state)).mean()
    #todo Why are we detaching here?

    next_q_values = target_value(next_state, target_policy(next_state).detach())
    q_vals = online_value(state, action)

    val_loss = ((reward + gamma * next_q_values.squeeze() * (1 - done) - q_vals.squeeze()) ** 2).mean()


    policy_opt.zero_grad()
    policy_loss.backward()
    policy_opt.step()

    value_opt.zero_grad()
    val_loss.backward()
    value_opt.step()

    for target_param, param in zip(target_value.parameters(), online_value.parameters()):
        target_param.data.mul_(polyak)
        target_param.data.add_(param.data*(1-polyak))


    for target_param, param in zip(target_policy.parameters(), online_policy.parameters()):
        target_param.data.mul_(polyak)
        target_param.data.add_(param.data*(1-polyak))


env = gym.make("Pendulum-v1")
ou_noise = OUNoise(env.action_space)

#Can't understand the state dimension of inverted pendulum
dim_state = env.observation_space.shape[0]
dim_action = env.action_space.shape[0]
act_limit = env.action_space.high[0]
online_value = ValueNetwork(dim_state, dim_action).to(device)
online_policy = PolicyNetwork(dim_state, dim_action).to(device)
target_value = ValueNetwork(dim_state, dim_action).to(device)
target_policy = PolicyNetwork(dim_state, dim_action).to(device)

for target_param, online_param in zip(target_value.parameters(), online_value.parameters()):
    target_param.data.copy_(online_param.data)
for target_param, online_param in zip(target_policy.parameters(), online_policy.parameters()):
    target_param.data.copy_(online_param.data)
value_opt = optim.Adam(online_value.parameters(), lr=value_lr)
policy_opt = optim.Adam(online_policy.parameters(), lr = policy_lr)

def add_exploration(action, iteration):
    action += (noise_scale/iteration)*np.random.randn(dim_action)
    return np.clip(action, -act_limit, act_limit)

tot_rewards = []
frame_index = 0
for i in range(episodes):
    state = torch.tensor(env.reset(), dtype=torch.float32)
    ou_noise.reset()

    done = False
    eps_rew = 0
    steps = 0
    while not done:
        print("frame_index = ", frame_index, "episode = ", i)
        action = online_policy(state.to(device)).cpu().detach().numpy()
        # action = add_exploration(action, steps+1)
        action = ou_noise.get_action(action, steps)

        print("actions = ", action)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, next_state, reward, done, action))
        if len(replay_buffer)>batch_size:
            ddpg_update(batch_size)
        eps_rew += reward
        state = torch.tensor(next_state, dtype=torch.float32)
        steps += 1
        frame_index += 1

        if done:
            tot_rewards.append(eps_rew)
            break

    if (i % 20) == 0:
        plt.plot(tot_rewards)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        np.savetxt("tot_rewards_ddpg.csv", np.array(tot_rewards), delimiter=' ', fmt='%s')


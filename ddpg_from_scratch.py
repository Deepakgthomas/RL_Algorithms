# Modifying the code given over here - https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb

import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import random
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
value_lr = 1e-3
policy_lr = 1e-4
mem_size = 1000000
replay_buffer = deque(maxlen=mem_size)
episodes = 50000
batch_size = 10000
gamma = 0.99
class ValueNetwork(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(ValueNetwork,self).__init__()
        self.linear1 = nn.Linear(dim_state+dim_action, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32,1)

    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(PolicyNetwork,self).__init__()
        self.linear1 = nn.Linear(dim_state, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, dim_action)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x


def ddpg_update(batch_size):
    state, next_state, reward, done, action = zip(*random.sample(replay_buffer, batch_size))
    state = torch.stack(list(state), dim=0).squeeze(1)
    # state= state.reshape(batch_size, 3, 210, 160).to(device)
    state = state.to(device)
    # next_state = torch.from_numpy(np.array(next_state)).reshape(batch_size, 3, 210, 160).type(torch.float32).to(device)
    next_state = torch.from_numpy(np.array(next_state)).type(torch.float32).to(device)

    reward = torch.from_numpy(np.array(reward)).to(device)

    done = torch.from_numpy(np.array(done)).long().to(device)
    action = torch.from_numpy(np.array(action)).type(torch.int64).to(device)

    policy_loss = -value(state, policy(state)).mean()
    #todo Why are we detaching here?

    next_q_values = value(next_state, policy(next_state).detach())
    q_vals = value(state, policy(state).detach())

    max_next_q_values = torch.max(next_q_values, -1)[0].detach()

    val_loss = ((reward + gamma * max_next_q_values * (1 - done) - q_vals.squeeze()) ** 2).mean()

    policy_opt.zero_grad()
    policy_loss.backward()
    policy_opt.step()

    value_opt.zero_grad()
    val_loss.backward()
    value_opt.step()

env = gym.make("Pendulum-v1")
#Can't understand the state dimension of inverted pendulum
dim_state = env.observation_space.shape[0]
print("Dimension of state = ", dim_state)
dim_action = env.action_space.shape[0]

value = ValueNetwork(dim_state, dim_action).to(device)
policy = PolicyNetwork(dim_state, dim_action).to(device)

value_opt = optim.Adam(value.parameters(), lr=value_lr)
policy_opt = optim.Adam(policy.parameters(), lr = policy_lr)

tot_rewards = []
frame_index = 0
for i in range(episodes):
    state = torch.tensor(env.reset(), dtype=torch.float32)
    episode_reward = 0
    done = False
    eps_rew = 0
    steps = 0
    while not done:
        print("frame_index = ", frame_index, "episode = ", i)
        action = policy(state.to(device)).cpu().detach().numpy()
        print("action = ", action)
        # print("env.step(action) = ", env.step([0.2]))
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, next_state, reward, done, action))
        if len(replay_buffer)>batch_size:
            ddpg_update(batch_size)
        eps_rew += reward
        state = torch.tensor(next_state, dtype=torch.float32)
        episode_reward += reward
        steps += 1
        frame_index += 1

        if done:
            tot_rewards.append(eps_rew)
            break

    if (i % 10) == 0:
        np.savetxt("tot_rewards_ddpg.csv", np.array(tot_rewards), delimiter=' ', fmt='%s')


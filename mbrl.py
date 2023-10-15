import numpy as np
import gymnasium as gym
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
import random
import torch

class Model(nn.Module):
    def __init__(self, state_size, action_size):
        super(Model,self).__init__()
        self.linear1 = nn.Linear(state_size, 100)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(100, reward_space)
        self.linear3 = nn.Linear(100, state_size)
    def forward(self,x ):

        x = self.linear1(x)
        x = self.relu(x)
        reward = self.linear2(x)
        next_state = self.linear3(x)
        return reward, next_state

env = gym.make("CliffWalking-v0")
mem_size = 5000
episodes = 500
eps = 1.0
learning_rate = 0.1
discount_factor = 0.99
reward_space = 1

replay_buffer = deque(maxlen=mem_size)



def MBRL(eps):
    tot_rewards = []
    buffer = []

    n_iters = 500
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    model = Model(state_size= np.array(1),action_size=np.array(1))
    opt = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    for i in range(episodes):
        print("episode = ", i)
        state = env.reset()[0]
        done = False
        steps = 0
        eps_reward = 0
        while not done and steps < 50:
            if np.random.uniform(0, 1) < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            next_state, reward, terminated, truncated, info = env.step(action)
            buffer.append((state, action))
            Q[state, action] = Q[state, action] + learning_rate * (
                        reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

            replay_buffer.append((state, action, reward, next_state))
            eps = eps / (1 + 0.001)
            eps_reward += reward
            if terminated:
                break
            state = next_state
            steps += 1
        tot_rewards.append(eps_reward)
        for _ in range(n_iters):

            state, action, reward, next_state= zip(*random.sample(replay_buffer, 1))
            tensor_state = torch.tensor(np.array(state[0]).reshape(1, -1), dtype=torch.float32)
            sampled_reward,  sampled_next_state = model(tensor_state)
            if(int(sampled_next_state) == next_state):
                loss_state = 0
            else:
                loss_state = 1
            loss = loss_state + (sampled_reward.detach().numpy()[0][0] - reward[0])
            loss = torch.tensor(np.array(loss), requires_grad=True)
            opt.zero_grad()
            loss.backward()
            opt.step()

            Q[state, action] = Q[state, action] + learning_rate * (
                        sampled_reward.detach().numpy() + discount_factor * np.max(Q[int(sampled_next_state.detach().numpy()), :]) - Q[
                    state, action])

    return tot_rewards




dyna_returns = MBRL(eps)
plt.plot(dyna_returns, label='dyna')
plt.legend()
plt.show()
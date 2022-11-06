
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
import sys
env = gym.make("PongNoFrameskip-v4")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Hyperparameters
episodes = 200000
eps = 1.0
learning_rate = 0.0001
tot_rewards = []
tot_loss = []
decay_val = 0.0001
mem_size = 20000
batch_size = 128
gamma = 0.99
update_target = 200
max_steps = 200
PATH = "./saved_models/pong_batch_size_352"

class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(NeuralNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.grayscale = tv.transforms.Grayscale()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(13824, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, action_size)

    def forward(self, x):
        x = self.grayscale(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



print(env.observation_space.shape[-1])
model = NeuralNetwork(env.observation_space.shape[-1], env.action_space.n).to(device)
target = NeuralNetwork(env.observation_space.shape[-1], env.action_space.n).to(device)

opt = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
replay_buffer = deque(maxlen=mem_size)


def compute_td_loss(batch_size):
    state, next_state, reward, done, action = zip(*random.sample(replay_buffer, batch_size))
    state = torch.stack(list(state), dim=0).squeeze(1)
    state= state.reshape(batch_size, 3, 210, 160).to(device)
    next_state = torch.from_numpy(np.array(next_state)).reshape(batch_size, 3, 210, 160).type(torch.float32).to(device)
    reward = torch.from_numpy(np.array(reward)).to(device)
    done = torch.from_numpy(np.array(done)).long().to(device)
    action = torch.from_numpy(np.array(action)).type(torch.int64).to(device)
    q_values = model(state)
    next_q_values = target(next_state)
    q_vals = q_values.gather(dim=-1, index=action.reshape(-1,1))
    max_next_q_values = torch.max(next_q_values,-1)[0].detach()
    loss = ((reward + gamma*max_next_q_values*(1-done) - q_vals.squeeze())**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss

if os.path.exists(PATH):
    model.load_state_dict(torch.load(PATH))
else:
    frame_index = 0
    for i in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
        state= state.reshape(1, 3, 210, 160)
        done = False
        steps = 0
        eps_rew = 0 
        eps_loss = 0
        while not done:
            print("frame_index = ", frame_index, "episode = ", i)
            if np.random.uniform(0,1)<eps:
                action = env.action_space.sample()
            else:
                action = torch.argmax(model(state.to(device))).cpu().detach().numpy()


            next_state, reward, done, info = env.step(action)
            replay_buffer.append((state, next_state, reward, done, action))
            if len(replay_buffer)==mem_size:
                loss = compute_td_loss(batch_size)
                eps_loss += loss.cpu().detach().numpy()
            eps = eps/(1 + decay_val)
            eps_rew += reward 

            if steps%update_target==0:
                target.load_state_dict(model.state_dict())

            if done:
                tot_rewards.append(eps_rew)
                break

            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            state= state.reshape(1, 3, 210, 160)
            steps += 1
            frame_index += 1
        tot_rewards.append(eps_rew)
        tot_loss.append(eps_loss)

        if(i%10)==0:
            np.savetxt("tot_rewards.csv", np.array(tot_rewards), delimiter=' ', fmt='%s')
            torch.save(model.state_dict(), PATH)
    torch.save(model.state_dict(), PATH)


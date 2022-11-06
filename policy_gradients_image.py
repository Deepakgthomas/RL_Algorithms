#I referred to this when writing the code - https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Chapter%204/Ch4_book.ipynb

import numpy as np
import gym
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision as tv
import torch.nn.functional as F
env = gym.make("ALE/Pong-v5")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
episodes = 10000

def discount_rewards(reward, gamma = 0.99):
    # return torch.pow(gamma, torch.arange(len(reward)))*reward
    R = 0
    returns = []
    reward = reward.tolist()
    for r in reward[::-1]:
        R = r + gamma * R
        returns.append(R)

    returns = torch.tensor(returns[::-1])
    return returns
def normalize_rewards(disc_reward):
    if disc_reward.max()!=0:
        return disc_reward/(disc_reward.max())
    else:
        return disc_reward / (disc_reward.max()+0.001)

class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(NeuralNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.grayscale = tv.transforms.Grayscale()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(13824, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, action_size)
        self.softmax = nn.Softmax()

    def forward(self,x):
        x = self.grayscale(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

model = NeuralNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
opt = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
score = []
for i in range(episodes):
    print("i = ", i)
    state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
    state = state.reshape(1, 3, 210, 160)
    done = False
    transitions = []

    tot_rewards = 0
    while not done:

        act_proba = model(state.to(device))
        # print("act_proba = ",act_proba)
        action = np.random.choice(np.array([0,1,2,3,4,5]), p = act_proba.cpu().data.numpy().reshape(-1))
        next_state, reward, done, info = env.step(action)
        tot_rewards += reward
        transitions.append((state, action, tot_rewards))
        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        state = state.reshape(1, 3, 210, 160)


    if i>0 and i%50==0:
        print("i = ", i, ", reward = ", tot_rewards, ", loss = ", loss)
    score.append(tot_rewards)
    reward_batch = torch.Tensor([r for (s,a,r) in transitions])
    disc_rewards = discount_rewards(reward_batch)
    # print("disc_rewards = ", disc_rewards)
    # nrml_disc_rewards = normalize_rewards(disc_rewards).to(device)
    nrml_disc_rewards = disc_rewards.to(device)
    state_batch = [s for (s,a,r) in transitions]
    state_batch = torch.stack(state_batch).reshape(-1,3,210, 160)
    action_batch = torch.Tensor([a for (s,a,r) in transitions]).to(device)
    pred_batch = model(state_batch.to(device))
    print("pred_batch = ", pred_batch, "action_batch = ", action_batch)
    prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()
    # print("prob_batch = ", torch.log(prob_batch))
    loss = -(torch.sum(torch.log(prob_batch)*nrml_disc_rewards))
    # print("loss = ", loss)
    opt.zero_grad()
    loss.backward()
    opt.step()

plt.scatter(np.arange(len(score)), score)
plt.show()







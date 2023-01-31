#Modified this code - https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Chapter%204/Ch4_book.ipynb

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
def normalize_rewards(disc_reward):
    return disc_reward/(disc_reward.max())

class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(NeuralNetwork, self).__init__()
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

model = NeuralNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
opt = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
score = []
for i in range(episodes):
    print("i = ", i)
    state = env.reset()
    done = False
    transitions = []

    tot_rewards = 0
    while not done:

        act_proba = model(torch.from_numpy(state).to(device))
        action = np.random.choice(np.array([0,1]), p = act_proba.cpu().data.numpy())
        next_state, reward, done, info = env.step(action)
        tot_rewards += 1
        transitions.append((state, action, tot_rewards))
        state = next_state


    if i%50==0:
        print("i = ", i, ",reward = ", tot_rewards)
    score.append(tot_rewards)
    reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims = (0,))

    disc_rewards = discount_rewards(reward_batch)
    nrml_disc_rewards = normalize_rewards(disc_rewards).to(device)
    state_batch = torch.Tensor([s for (s,a,r) in transitions])
    action_batch = torch.Tensor([a for (s,a,r) in transitions]).to(device)
    pred_batch = model(state_batch.to(device))
    # print("pred_batch ", pred_batch)
    prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()
    # print("prob_batch = ", prob_batch)
    loss = -(torch.sum(torch.log(prob_batch)*nrml_disc_rewards))
    opt.zero_grad()
    loss.backward()
    opt.step()

plt.scatter(np.arange(len(score)), score)
plt.show()







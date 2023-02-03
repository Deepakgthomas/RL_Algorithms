#Modified this code - https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Chapter%204/Ch4_book.ipynb
#Also, modified this code - https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
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



actor = Actor(env.observation_space.shape[0], env.action_space.n).to(device)
critic = Critic(env.observation_space.shape[0], env.action_space.n).to(device)
policy_opt = torch.optim.Adam(params = actor.parameters(), lr = learning_rate)
value_opt = torch.optim.Adam(params = critic.parameters(), lr = learning_rate)

score = []
for i in range(episodes):
    print("i = ", i)
    state = env.reset()
    done = False
    transitions = []

    tot_rewards = 0
    while not done:

        value = critic(torch.from_numpy(state).to(device))
        policy = actor(torch.from_numpy(state).to(device))
        action = np.random.choice(np.array([0, 1]), p=policy.cpu().data.numpy())
        next_state, reward, done, info = env.step(action)
        tot_rewards += 1
        transitions.append((state, action, tot_rewards, next_state))
        state = next_state



    if i%50==0:
        print("i = ", i, ",reward = ", tot_rewards)
    score.append(tot_rewards)
    reward_batch = torch.Tensor([r for (s,a,r, ns) in transitions]).flip(dims = (0,))

    disc_rewards = discount_rewards(reward_batch)
    nrml_disc_rewards = normalize_rewards(disc_rewards).to(device)
    state_batch = torch.Tensor([s for (s,a,r, ns) in transitions]).to(device)
    action_batch = torch.Tensor([a for (s,a,r, ns) in transitions]).to(device)
    next_state_batch = torch.Tensor([ns for (s,a,r, ns) in transitions]).to(device)

    pred_batch = actor(state_batch)
    prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()
    values = critic(state_batch)


    advantage = nrml_disc_rewards-values
    critic_loss = advantage.pow(2).mean()
    actor_loss = -(torch.sum(torch.log(prob_batch)*nrml_disc_rewards))


    policy_opt.zero_grad()
    actor_loss.backward()
    policy_opt.step()

    value_opt.zero_grad()
    critic_loss.backward()
    value_opt.step()

    if i%50==0:
        plt.scatter(np.arange(len(score)), score)
        plt.show(block=False)
        plt.pause(3)
        plt.close()

plt.scatter(np.arange(len(score)), score)
plt.show()







#Modified this code - https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Chapter%204/Ch4_book.ipynb
#Also, modified this code - https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
# Also, modified this code - https://github.com/ericyangyu/PPO-for-Beginners/blob/9abd435771aa84764d8d0d1f737fa39118b74019/ppo.py#L151
import numpy as np
import gym
import torch
from torch import nn
import matplotlib.pyplot as plt
env = gym.make('Pendulum-v1')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 0.0001
episodes = 10000
gamma = 0.99

clip = 0.2
dim_action = env.action_space.shape[0]

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


def rollout():


    for i in range(5): # 100 episodes should be good?
        # obs = env.reset()
        print("Rollout process, i = ", i)
        obs = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)

        tot_rewards = 0
        transitions = []
        iter = 0
        done = False
        while not done:
            act_probs = torch.distributions.Categorical(actor(obs.to(device)))
            action = act_probs.sample()
            action = action.cpu().detach().numpy()
            next_state, reward, done, info = env.step(action)
            action = torch.tensor(action, dtype=torch.float32).to(device)

            tot_rewards += np.power(gamma, iter) * reward

            iter += 1
            transitions.append((obs, action, act_probs.log_prob(action), tot_rewards))
            obs = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        print("Discounted Reward = ", tot_rewards)
    batch_obs = torch.Tensor([s.numpy() for (s, a, a_p, r) in transitions]).to(device)
    # print("batch_obs shape = ", np.array(batch_obs).shape)
    batch_act = torch.Tensor([a for (s, a, a_p, r) in transitions]).to(device)
    batch_log_probs = torch.Tensor([a_p for (s, a, a_p, r) in transitions]).to(device)
    batch_rtgs = torch.Tensor([r for (s, a, a_p, r) in transitions]).flip(dims = (0,)).to(device)

    return batch_obs, batch_act, batch_log_probs, batch_rtgs

actor = Actor(env.observation_space.shape[0], dim_action).to(device)
critic = Critic(env.observation_space.shape[0], dim_action).to(device)
policy_opt = torch.optim.Adam(params = actor.parameters(), lr = learning_rate)
value_opt = torch.optim.Adam(params = critic.parameters(), lr = learning_rate)

score = []
for i in range(episodes):
    batch_obs, batch_act, batch_log_probs, batch_rtgs = rollout()
    value = critic(batch_obs)
    batch_rtgs = batch_rtgs
    # todo Why are we detaching value
    A_k = batch_rtgs - value.detach()

    for _ in range(10):
        value = critic(batch_obs)
        act_probs = torch.distributions.Categorical(actor(batch_obs))

        action = act_probs.sample()
        log_probs = act_probs.log_prob(action)

        ratios = torch.exp(log_probs - batch_log_probs)
        surr1 = ratios*A_k
        surr2 = torch.clamp(ratios, 1 - clip, 1 + clip)*A_k

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (value - batch_rtgs).pow(2).mean()
        #todo No idea why we are doing retain_graph = True
        policy_opt.zero_grad()
        actor_loss.backward(retain_graph=True)
        policy_opt.step()

        value_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        value_opt.step()




















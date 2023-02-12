#Modified this code - https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Chapter%204/Ch4_book.ipynb
#Also, modified this code - https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
# Also, modified this code - https://github.com/ericyangyu/PPO-for-Beginners/blob/9abd435771aa84764d8d0d1f737fa39118b74019/ppo.py#L151
import numpy as np
import gym
import torch
from torch import nn
torch.manual_seed(798)
import matplotlib.pyplot as plt
env = gym.make('Acrobot-v1')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 0.05
episodes = 10000
gamma = 0.99
clip = 0.2

#No idea whether these hyperparameters are good
ppo_batch = 30
training_iters = 5


# dim_action = env.action_space.shape[0]

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax())
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
    transitions = []
    disc_reward_list = []
    for i in range(ppo_batch):
        print("Rollout process, i = ", i)
        obs = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)

        all_rewards = []

        iter = 0
        done = False
        tot_rewards = 0
        while not done:
            act_probs = torch.distributions.Categorical(actor(obs.to(device)))
            # print("act_probs = ", actor(obs.to(device)))
            action = act_probs.sample().squeeze()
            action = action.cpu().detach().numpy()
            next_state, reward, done, info = env.step(action)
            action = torch.tensor(action, dtype=torch.float32).to(device)
            all_rewards.append(reward)
            tot_rewards += reward
            iter += 1
            transitions.append((obs, action, act_probs.log_prob(action)))
            obs = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        print("Reward = ", tot_rewards)
        eps_rew = 0
        eps_rew_list = []
        for reward in reversed(all_rewards):
            eps_rew = eps_rew*gamma + reward
            eps_rew_list.append(eps_rew)

        for rtgs in reversed(eps_rew_list):
            disc_reward_list.append(rtgs)
    batch_obs = torch.Tensor([s.numpy() for (s, a, a_p) in transitions]).to(device)
    batch_act = torch.Tensor([a for (s, a, a_p) in transitions]).to(device)
    batch_log_probs = torch.Tensor([a_p for (s, a, a_p) in transitions]).to(device)
    batch_rtgs = torch.Tensor(disc_reward_list).to(device)

    return batch_obs, batch_act, batch_log_probs, batch_rtgs

actor = Actor(env.observation_space.shape[0], 1).to(device)
critic = Critic(env.observation_space.shape[0], 1).to(device)
policy_opt = torch.optim.Adam(params = actor.parameters(), lr = learning_rate)
value_opt = torch.optim.Adam(params = critic.parameters(), lr = learning_rate)

score = []
for i in range(episodes):
    batch_obs, batch_act, batch_log_probs, batch_rtgs = rollout()

    value = critic(batch_obs)
    # todo Why are we detaching value
    A_k = batch_rtgs - value.squeeze().detach()
    A_k = (A_k - A_k.mean())/A_k.std() + 1e-8

    for _ in range(training_iters):
        value = critic(batch_obs).squeeze()
        assert(value.ndim==1)
        policy = actor(batch_obs).squeeze()

        act_probs = torch.distributions.Categorical(policy)
        log_probs = act_probs.log_prob(batch_act).squeeze()
        # print("log_probs = ", log_probs)


        ratios = torch.exp(log_probs - batch_log_probs)
        assert(ratios.ndim==1)
        # print("ratios = ", ratios.shape)
        surr1 = ratios*A_k
        assert (surr1.ndim == 1)
        surr2 = torch.clamp(ratios, 1 - clip, 1 + clip)*A_k
        assert (surr2.ndim == 1)
        actor_loss = -torch.min(surr1, surr2).mean()
        print("actor_loss = ", actor_loss)
        critic_loss = (value - batch_rtgs).pow(2).mean()
        print("critic_loss = ", critic_loss)


        #todo No idea why we are doing retain_graph = True
        policy_opt.zero_grad()
        actor_loss.backward(retain_graph=True)
        policy_opt.step()

        value_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        value_opt.step()




















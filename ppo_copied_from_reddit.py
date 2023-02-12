import numpy as np
import gym
import torch
from torch import nn


import matplotlib.pyplot as plt

if gym.__version__ < '0.26':
    env = gym.make('Acrobot-v1', new_step_api=True)
else:
    env = gym.make('Acrobot-v1')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 2.5e-4
episodes = 10000
gamma = 0.99

clip = 0.2

# No idea whether these hyperparameters are good
ppo_batch = 30
training_iters = 4

# dim_action = env.action_space.shape[0]
dim_action = env.action_space.n


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            # nn.Linear(300, 128),
            # nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_stack = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            # nn.Linear(300, 128),
            # nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.linear_stack(x)
        return x


def rollout():
    transitions = []
    rtgs_list = []
    for i in range(5):  # 100 episodes should be good?
        print("Rollout process, i = ", i)
        # obs = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        tot_rewards = 0

        #### SERIOUSLY why are we emptying the data it should be initialised before the for loop?
        # transitions = []
        iter = 0
        done = False
        trunc = False
        rewards = []
        with torch.no_grad():
            while not done and not trunc:
                # obs_tensor uses obs instead of next_state
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act_probs = torch.distributions.Categorical(actor(obs_tensor))
                # act_probs = torch.distributions.Categorical(actor(obs.to(device)))
                action = act_probs.sample()

                ## action in device , use it to calculate log_prob before moving it to cpu
                log_prob = act_probs.log_prob(action)
                log_prob = log_prob.cpu().numpy()
                # no need to detach now
                # action = action.cpu().detach().numpy()
                # action = action.cpu().numpy()
                action = action.cpu().numpy()[0]  # take first action from a list that contains only 1 action :S
                # next_state, reward, done, info = env.step(action)
                next_state, reward, done, trunc, info = env.step(action)
                # action = torch.tensor(action, dtype=torch.float32).to(device)

                ##### CRITICAL
                # rewards to go needs future rewards ,not past rewards
                # tot_rewards += np.power(gamma, iter) * reward
                tot_rewards += reward
                iter += 1

                # we do not need the total_reward
                # transitions.append((obs, action, log_prob, tot_rewards))
                rewards.append(reward)
                # add the reward instead to calculate rtgs
                transitions.append((obs, action, log_prob))
                # added this to let our next_State be our state
                obs = next_state

        reversed_rtgs = []
        reverse_rtg = 0
        for r in reversed(rewards):
            reverse_rtg = reverse_rtg * gamma + r
            reversed_rtgs.append(reverse_rtg)

        for rtg in reversed(reversed_rtgs):
            rtgs_list.append(rtg)
        print("Episode Reward = ", tot_rewards)

    # d = zip(transitions)
    obs_ar, act_ar, log_probs_ar = list(zip(*transitions))
    rtgs_array = np.array(rtgs_list)

    # batch_obs = torch.Tensor([s.numpy() for (s, a, a_p, r) in transitions]).to(device)
    # # print("batch_obs shape = ", np.array(batch_obs).shape)
    # batch_act = torch.Tensor([a for (s, a, a_p, r) in transitions]).to(device)
    # batch_log_probs = torch.Tensor([a_p for (s, a, a_p, r) in transitions]).to(device)
    # # batch_rtgs = torch.Tensor([r for (s, a, a_p, r) in transitions]).flip(dims = (0,)).to(device)

    batch_obs = torch.tensor(obs_ar, dtype=torch.float32, device=device)
    batch_act = torch.tensor(act_ar, dtype=torch.int32, device=device).squeeze()
    batch_log_probs = torch.tensor(log_probs_ar, dtype=torch.float32, device=device).squeeze()
    batch_rtgs = torch.tensor(rtgs_array, dtype=torch.float32, device=device).squeeze()
    return batch_obs, batch_act, batch_log_probs, batch_rtgs


actor = Actor(env.observation_space.shape[0], dim_action).to(device)
critic = Critic(env.observation_space.shape[0], dim_action).to(device)
policy_opt = torch.optim.Adam(params=actor.parameters(), lr=learning_rate)
value_opt = torch.optim.Adam(params=critic.parameters(), lr=learning_rate)

score = []
for i in range(episodes):
    all_obs, all_actions, all_log_probs, all_rtgs = rollout()
    # if we do not need grad , then torch.no_grad is faster and use less memory "don't quote me on that lol"
    with torch.no_grad():
        # no need to detach now no grads
        value = critic(all_obs)

    # no need to detach because we calulated value using no_grad settings
    all_A_k = all_rtgs - value.squeeze()
    # normalizing the advantage is a good thing you can skip it thought
    all_A_k = (all_A_k - all_A_k.mean()) / all_A_k.std() + 1e-8

    # NOTE use detach if you calculated value without using no_grad
    # todo Why are we detaching value ? detach returns same tensor but without grads , so when calling backward and step it won't change the critic which is used to calculate the value we are using the advantage to optimize the actor , not the other way around
    # A_k = batch_rtgs - value.squeeze().detach()
    batch_size = len(all_obs) // training_iters
    for _ in range(4):
        indices = torch.randint(len(all_obs), size=(batch_size,))
        batch_obs = all_obs[indices]
        batch_actions = all_actions[indices]
        batch_log_probs = all_log_probs[indices]
        batch_rtgs = all_rtgs[indices]
        batch_A_k = all_A_k[indices]
        for _ in range(training_iters):
            value = critic(batch_obs).squeeze()
            act_probs = torch.distributions.Categorical(actor(batch_obs))

            action = act_probs.sample()
            log_probs = act_probs.log_prob(batch_actions).squeeze()
            ratios = torch.exp(log_probs - batch_log_probs)
            surr1 = ratios * batch_A_k
            surr2 = torch.clamp(ratios, 1 - clip, 1 + clip) * batch_A_k

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (value - batch_rtgs).pow(2).mean()

            # todo No idea why we are doing retain_graph = True
            policy_opt.zero_grad()
            actor_loss.backward(retain_graph=True)
            policy_opt.step()

            value_opt.zero_grad()
            critic_loss.backward(retain_graph=True)
            value_opt.step()

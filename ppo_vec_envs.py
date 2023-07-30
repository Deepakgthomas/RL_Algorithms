#Modified this code - https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Chapter%204/Ch4_book.ipynb
#Also, modified this code - https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
# Also, modified this code - https://github.com/ericyangyu/PPO-for-Beginners/blob/9abd435771aa84764d8d0d1f737fa39118b74019/ppo.py#L151
if __name__ == '__main__':

    import numpy as np
    import gym
    import torch
    import random

    from torch import nn

    torch.manual_seed(798)
    import matplotlib.pyplot as plt
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    num_envs = 8
    env = gym.vector.make('Acrobot-v1', num_envs=num_envs)
    env.seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 2.5e-4
    episodes = 10000
    gamma = 0.99
    clip = 0.2

    #No idea whether these hyperparameters are good
    ppo_batch = 60
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
                nn.Linear(128, action_size),
                nn.Softmax(dim=-1))
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
    actor = Actor(env.observation_space.shape[1], env.action_space[0].n).to(device)
    critic = Critic(env.observation_space.shape[1], 1).to(device)
    policy_opt = torch.optim.Adam(params = actor.parameters(), lr = learning_rate)
    value_opt = torch.optim.Adam(params = critic.parameters(), lr = learning_rate)
    obs = torch.tensor(env.reset(), dtype=torch.float32).to(device)
    def rollout(obs):
        transitions = []
        disc_reward_list = []
        all_rewards = []
        tot_rewards = np.array([0 for i in range(8)], dtype=float)
        all_actions = []
        all_actions_probs = []
        all_observations = []
        all_dones = []
        # dones = np.array([0 for i in range(8)])
        for i in range(ppo_batch):
            act_probs = torch.distributions.Categorical(actor(obs.to(device)).squeeze())
            action = act_probs.sample().squeeze()
            action = action.cpu().detach().numpy()

            next_state, reward, done, info = env.step(action)

            action = torch.tensor(action, dtype=torch.float32).to(device)
            all_rewards.append(reward)
            tot_rewards[done] = 0
            tot_rewards += reward
            all_dones.append(done)
            all_observations.append(obs.cpu().detach().numpy().reshape(-1))
            all_actions.append(action.cpu().detach().numpy())
            all_actions_probs.append(act_probs.log_prob(action).cpu().detach().numpy())

            obs = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        print("total reward = ", tot_rewards)
        eps_rew = critic(obs.to(device)).cpu().detach().numpy().reshape(num_envs)
        eps_rew_list = []

        for reward, done in zip(reversed(all_rewards), reversed(all_dones)):

            eps_rew[done] = 0
            eps_rew = eps_rew*gamma + reward
            eps_rew_list.append(eps_rew)

        for rtgs in reversed(eps_rew_list):
            disc_reward_list.append(rtgs)
        batch_obs = torch.Tensor(all_observations).reshape(-1,env.observation_space.shape[1]).to(device)
        batch_act = torch.Tensor(np.array(all_actions).reshape(-1)).to(device)
        batch_log_probs = torch.Tensor(np.array(all_actions_probs).reshape(-1)).to(device)

        batch_rtgs = torch.Tensor(disc_reward_list).reshape(-1).to(device)

        return batch_obs, batch_act, batch_log_probs, batch_rtgs, obs
    # print("stuff = ",env.action_space[0].n)
    # print("stuff = ",env.observation_space.shape[1])



    score = []
    for i in range(episodes):
        print("i = ", i)
        batch_obs, batch_act, batch_log_probs, batch_rtgs, obs = rollout(obs)
        value = critic(batch_obs).squeeze()


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
            critic_loss = (value - batch_rtgs).pow(2).mean()


            #todo No idea why we are doing retain_graph = True
            policy_opt.zero_grad()
            actor_loss.backward(retain_graph=True)
            policy_opt.step()

            value_opt.zero_grad()
            critic_loss.backward(retain_graph=True)
            value_opt.step()



















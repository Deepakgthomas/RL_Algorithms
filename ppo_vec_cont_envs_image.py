#Modified this code - https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Chapter%204/Ch4_book.ipynb
#Also, modified this code - https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
# Also, modified this code - https://github.com/ericyangyu/PPO-for-Beginners/blob/9abd435771aa84764d8d0d1f737fa39118b74019/ppo.py#L151
# Got a lot of help from the subreddit - reinforcement_learning

if __name__ == '__main__':

    import numpy as np
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing
    import torch
    import random
    import matplotlib.pyplot as plt
    from torch import nn
    import torchvision as tv
    import torch.nn.functional as F
    torch.manual_seed(798)
    import matplotlib.pyplot as plt
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    from collections import deque
    num_envs = 12
    ent_coeff = 0.03
    num_channels = 1
    batches = 4
    channels = 3
    learning_rate = 0.00025
    episodes = 1500
    gae_lambda = 0.95
    gamma = 0.99
    clip = 0.2
    rollout_steps = 200
    training_iters = 4


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = gym.vector.make("BreakoutNoFrameskip-v4", num_envs=num_envs,wrappers=AtariPreprocessing)
    actor_PATH = './actor_model' + 'breakout' + '.pt'
    critic_PATH = './critic_model ' + 'pong'+ '.pt'
    square_size = env.observation_space.shape[-1]

    class Actor(nn.Module):
        def __init__(self, state_size, action_size):
            super(Actor, self).__init__()
            self.state_size = state_size
            self.action_size = action_size
            self.conv1 = nn.Conv2d(num_channels, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.conv3 = nn.Conv2d(16, 32, 3)
            self.fc1 = nn.Linear(2048, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, action_size)
            self.last = nn.Softmax(dim=-1)

        def forward(self,x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            # Here is my attempt at the reparameterization trick :/
            mean = self.fc3(x)
            std = self.fc3(x)
            epsilon = torch.randn(self.action_size)
            action = mean + std*epsilon
            return action

    class Critic(nn.Module):
        def __init__(self, state_size, action_size):
            super(Critic, self).__init__()
            self.state_size = state_size
            self.action_size = action_size
            self.conv1 = nn.Conv2d(num_channels, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.conv3 = nn.Conv2d(16, 32, 3)
            self.fc1 = nn.Linear(2048, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 1)

        def forward(self, x):
            x = x.reshape(-1, 1, square_size, square_size)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    actor = Actor(env.observation_space.shape[-1], env.action_space[0].n).to(device)
    critic = Critic(env.observation_space.shape[-1], 1).to(device)
    policy_opt = torch.optim.Adam(params = actor.parameters(), lr = learning_rate)
    value_opt = torch.optim.Adam(params = critic.parameters(), lr = learning_rate)
    obs = torch.tensor(env.reset()[0], dtype=torch.float32).to(device)

    tot_rewards = np.array([0 for i in range(num_envs)], dtype=float)
    final_scores = []
    last_n_rewards = deque(maxlen=10)
    def rollout(obs): #todo Why can't the rollout function access it from outside?

        all_rewards = []
        all_actions = []
        all_actions_probs = []
        all_observations = []
        all_dones = []
        global tot_rewards #todo Why did I have to declare tot_rewards as global?

        for i in range(rollout_steps):
            obs = obs.reshape(num_envs, 1, square_size, square_size)
            action = actor(obs.to(device)).squeeze()
            # action = act_probs.sample().squeeze()
            action = action.cpu().detach().numpy()
            next_state, reward, done, truncated, info = env.step(action)
            action = torch.tensor(action, dtype=torch.float32).to(device)

            # These statistics help determine how well the agent is performing.
            tot_rewards += reward
            for reward_val, done_val in zip(tot_rewards, done):
                if done_val:
                    print("reward_val = ", reward_val)
                    last_n_rewards.append(reward_val)
                    final_scores.append(reward_val)
            tot_rewards[done] = 0

            all_rewards.append(reward)
            all_dones.append(done)
            all_observations.append(obs.cpu().detach().numpy())
            all_actions.append(action.cpu().detach().numpy())
            all_actions_probs.append(act_probs.log_prob(action).cpu().detach().numpy())
            obs = torch.tensor(next_state, dtype=torch.float32)

        # Computing values over here
        eps_rew = critic(obs.to(device)).cpu().detach().numpy().reshape(-1)
        eps_rew_list = []
        state_value_list = []
        for reward, done in zip(reversed(all_rewards), reversed(all_dones)):
            eps_rew[done] = 0
            eps_rew = eps_rew*gamma + reward
            eps_rew_list.append(eps_rew.copy())
        next_adv = np.array([0 for i in range(num_envs)], dtype=float)
        batch_obs = torch.Tensor(all_observations).reshape(-1, num_envs, square_size, square_size)
        for rtgs in reversed(eps_rew_list):
            state_value_list.append(rtgs)

        # Computing advantages over here, A = Q - V
        val_next_state = eps_rew.copy()
        inv_eps_adv_list = []
        for reward,done,obs in zip(reversed(all_rewards), reversed(all_dones), reversed(batch_obs)):
            next_adv[done] = 0
            val_next_state[done] = 0
            val_current_state = critic(obs.to(device)).cpu().detach().numpy().reshape(-1)
            delta = reward + gamma*val_next_state-val_current_state
            adv = delta + gae_lambda * gamma * next_adv
            inv_eps_adv_list.append(adv)
            next_adv = adv.copy()
            val_next_state = val_current_state.copy()
        final_adv_list = []
        for a in reversed(inv_eps_adv_list):
            final_adv_list.append(a)

        # Returning all the data from the rollout. `obs` needs to be returned because the episode might not be over
        # for some environment
        batch_obs = torch.Tensor(all_observations).reshape(-1,env.observation_space.shape[1]).to(device)
        batch_act = torch.Tensor(np.array(all_actions).reshape(-1)).to(device)
        batch_log_probs = torch.Tensor(np.array(all_actions_probs).reshape(-1)).to(device)
        batch_rtgs = torch.Tensor(state_value_list).reshape(-1).to(device)
        batch_advantages = torch.Tensor(final_adv_list).reshape(-1).to(device)
        return batch_obs, batch_act, batch_log_probs, batch_rtgs, batch_advantages, obs

    #Learning Phase
    for episode in range(episodes):
        print("episodes = ", episode)
        all_obs, all_act, all_log_probs, all_rtgs, all_advantages, obs = rollout(obs)
        all_obs = all_obs.reshape(-1, 1, square_size, square_size)

        assert (all_obs.shape == (rollout_steps*num_envs, num_channels, square_size, square_size))
        assert (all_act.shape == (rollout_steps*num_envs,))
        assert (all_log_probs.shape == (rollout_steps*num_envs,))
        assert (all_rtgs.shape == (rollout_steps*num_envs,))
        assert (all_advantages.shape == (rollout_steps*num_envs,))

        # Standardize all advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        for i in range(training_iters):
            print("Training Iteration = ", i)
            total_examples = num_envs * rollout_steps
            batch_size = total_examples // batches
            batch_starts = np.arange(0, total_examples, batch_size)
            indices = np.arange(total_examples, dtype=np.int32)
            np.random.shuffle(indices)

            for batch_start in batch_starts:
                batch_end = batch_start + batch_size
                batch_index = indices[batch_start:batch_end]
                batch_obs = all_obs[batch_index]
                batch_act = all_act[batch_index]
                batch_log_probs = all_log_probs[batch_index]
                batch_rtgs = all_rtgs[batch_index]
                batch_advantages = all_advantages[batch_index]

                value = critic(batch_obs).squeeze()
                assert(value.ndim==1)
                policy = actor(batch_obs)
                # act_probs = torch.distributions.Categorical(policy)
                batch_entropy = act_probs.entropy().mean()
                log_probs = act_probs.log_prob(batch_act).squeeze()
                ratios = torch.exp(log_probs - batch_log_probs)
                assert(ratios.ndim==1)
                surr1 = ratios*batch_advantages
                assert (surr1.ndim == 1)
                surr2 = torch.clamp(ratios, 1 - clip, 1 + clip)*batch_advantages
                assert (surr2.ndim == 1)
                actor_loss = -torch.min(surr1, surr2).mean()  - ent_coeff*batch_entropy
                critic_loss = (value - batch_rtgs).pow(2).mean()

                policy_opt.zero_grad()
                actor_loss.backward(retain_graph=True)
                policy_opt.step()

                value_opt.zero_grad()
                critic_loss.backward(retain_graph=True)
                value_opt.step()

        if episode % 100 == 0:
            print("Saved")
            torch.save(actor.state_dict(), actor_PATH)
            torch.save(critic.state_dict(), critic_PATH)

    plt.plot(final_scores)
    plt.show()




















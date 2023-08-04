#Modified this code - https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Chapter%204/Ch4_book.ipynb
#Also, modified this code - https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb
# Also, modified this code - https://github.com/ericyangyu/PPO-for-Beginners/blob/9abd435771aa84764d8d0d1f737fa39118b74019/ppo.py#L151
if __name__ == '__main__':

    import numpy as np
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStack, GrayScaleObservation, TransformObservation

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
    num_stack = 3
    batches = 30
    channels = 3
    def reshape_image(obs):
        new_obs = np.array(obs).reshape(num_envs, 210, 160, 3*num_stack)
        return new_obs
    env = gym.vector.make("Pong-v4", num_envs=num_envs)
    env = FrameStack(env, num_stack=num_stack)
    env = TransformObservation(env, reshape_image)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.00025
    episodes = 500
    gamma = 0.99
    clip = 0.2

    #No idea whether these hyperparameters are good
    rollout_steps = 50
    training_iters = 15


    # dim_action = env.action_space.shape[0]

    class Actor(nn.Module):
        def __init__(self, state_size, action_size):
            super(Actor, self).__init__()
            self.state_size = state_size
            self.action_size = action_size
            self.grayscale = tv.transforms.Grayscale()
            self.conv1 = nn.Conv2d(num_stack, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.conv3 = nn.Conv2d(16, 32, 3)
            self.fc1 = nn.Linear(13824, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, action_size)
            self.last = nn.Softmax(dim=-1)
        def forward(self,x):
            x = x.reshape(-1, 210, 160, num_stack, 3)
            x = x.permute(0, 3, 4, 2, 1)
            x = self.grayscale(x)
            x = x.reshape(-1, num_stack, 210, 160)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            x = self.last(x)
            return x

    class Critic(nn.Module):
        def __init__(self, state_size, action_size):
            super(Critic, self).__init__()
            self.state_size = state_size
            self.action_size = action_size
            self.grayscale = tv.transforms.Grayscale()
            self.conv1 = nn.Conv2d(num_stack, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.conv3 = nn.Conv2d(16, 32, 3)
            self.fc1 = nn.Linear(13824, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 1)

        def forward(self, x):
            x = x.reshape(-1, 210, 160, num_stack, 3)
            x = x.permute(0, 3, 4, 2, 1)

            x = self.grayscale(x)

            x = x.reshape(-1, num_stack, 210, 160)

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
    obs = torch.tensor(env.reset()[0], dtype=torch.float32).to(device) #Gymnasium is quite different from gym


    tot_rewards = np.array([0 for i in range(num_envs)], dtype=float)
    final_scores = []
    last_n_rewards = deque(maxlen=10)
    def rollout(obs): #Why can't the rollout function access it from outside?

        disc_reward_list = []
        all_rewards = []
        all_actions = []
        all_actions_probs = []
        all_observations = []
        all_dones = []
        global tot_rewards #Why did I have to declare tot_rewards as global?
        for i in range(rollout_steps):

            foo = actor(obs.to(device))
            act_probs = torch.distributions.Categorical(actor(obs.to(device)).squeeze())
            action = act_probs.sample().squeeze()
            action = action.cpu().detach().numpy()

            next_state, reward, done, truncated, info = env.step(action)

            action = torch.tensor(action, dtype=torch.float32).to(device)
            all_rewards.append(reward)
            tot_rewards += reward
            for reward_val, done_val in zip(tot_rewards, done):
                if done_val:
                    print("reward_val = ", reward_val)
                    last_n_rewards.append(reward_val)
                    final_scores.append(reward_val)
            tot_rewards[done] = 0
            all_dones.append(done)
            all_observations.append(obs.cpu().detach().numpy().reshape(-1))
            all_actions.append(action.cpu().detach().numpy())
            all_actions_probs.append(act_probs.log_prob(action).cpu().detach().numpy())

            obs = torch.tensor(next_state, dtype=torch.float32)
        eps_rew = critic(obs.to(device)).cpu().detach().numpy().reshape(num_envs)
        eps_rew_list = []

        for reward, done in zip(reversed(all_rewards), reversed(all_dones)):

            eps_rew[done] = 0
            eps_rew = eps_rew*gamma + reward
            eps_rew_list.append(eps_rew.copy())

        for rtgs in reversed(eps_rew_list):
            disc_reward_list.append(rtgs)
        batch_obs = torch.Tensor(all_observations).reshape(-1,env.observation_space.shape[1]).to(device)
        batch_act = torch.Tensor(np.array(all_actions).reshape(-1)).to(device)

        batch_log_probs = torch.Tensor(np.array(all_actions_probs).reshape(-1)).to(device)

        batch_rtgs = torch.Tensor(disc_reward_list).reshape(-1).to(device)


        return batch_obs, batch_act, batch_log_probs, batch_rtgs, obs

    for i in range(episodes):
        print("episodes = ", i)
        all_obs, all_act, all_log_probs, all_rtgs, obs = rollout(obs)
        all_obs = all_obs.reshape(-1,210,160,num_stack, channels)



        value = critic(all_obs).squeeze()

        # todo Why are we detaching value
        A_k = all_rtgs - value.squeeze().detach()
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-8)




        for _ in range(training_iters):

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

                batch_advantages = A_k[batch_index]

                value = critic(batch_obs).squeeze()
                assert(value.ndim==1)
                policy = actor(batch_obs)

                act_probs = torch.distributions.Categorical(policy)

                log_probs = act_probs.log_prob(batch_act).squeeze()

                ratios = torch.exp(log_probs - batch_log_probs)
                assert(ratios.ndim==1)
                surr1 = ratios*batch_advantages
                assert (surr1.ndim == 1)
                surr2 = torch.clamp(ratios, 1 - clip, 1 + clip)*batch_advantages
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

    plt.plot(final_scores)
    plt.show()




















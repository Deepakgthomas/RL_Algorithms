import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("CliffWalking-v0")
episodes = 500
eps = 1.0
learning_rate = 0.1
discount_factor = 0.99


def dyna(eps):
    tot_rewards = []
    buffer = []

    n_iters = 500
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    mod_reward = np.zeros((env.observation_space.n, env.action_space.n))
    mod_next_state = np.zeros((env.observation_space.n, env.action_space.n))

    for i in range(episodes):
        state = env.reset()[0]
        done = False
        steps = 0
        eps_reward = 0
        while not done and steps < 50:
            if np.random.uniform(0, 1) < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            next_state, reward, terminated, truncated, info = env.step(action)
            buffer.append((state, action))
            Q[state, action] = Q[state, action] + learning_rate * (
                        reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
            mod_reward[state, action] = reward
            mod_next_state[state, action] = next_state
            eps = eps / (1 + 0.001)
            eps_reward += reward
            if terminated:
                break
            state = next_state
            steps += 1
        tot_rewards.append(eps_reward)
        for _ in range(n_iters):
            rand_index = np.random.randint(0, len(buffer))
            rand_state_index = buffer[rand_index][0]
            rand_action_index = buffer[rand_index][1]
            sampled_reward = int(mod_reward[rand_state_index, rand_action_index])
            sampled_next_state = int(mod_next_state[rand_state_index, rand_action_index])

            Q[rand_state_index, rand_action_index] = Q[rand_state_index, rand_action_index] + learning_rate * (
                        sampled_reward + discount_factor * np.max(Q[sampled_next_state, :]) - Q[
                    rand_state_index, rand_action_index])
    return tot_rewards


def q_learning(eps):
    # Hyperparameters

    tot_rewards = []

    # Minimal Q learning example
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for i in range(episodes):
        state = env.reset()[0]
        done = False
        steps = 0
        eps_rew = 0
        while not done and steps < 50:
            if np.random.uniform(0, 1) < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            next_state, reward, terminated, truncated, info = env.step(action)
            Q[state, action] = Q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
            eps = eps / (1 + 0.001)
            eps_rew += reward
            if terminated or truncated:
                break
            state = next_state
            steps += 1
        tot_rewards.append(eps_rew)

    return tot_rewards


dyna_returns = dyna(eps)
q_returns = q_learning(eps)
plt.plot(dyna_returns, label='dyna')
plt.plot(q_returns, label='q_learning')
plt.legend()
plt.show()
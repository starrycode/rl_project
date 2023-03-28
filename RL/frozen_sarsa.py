import random

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v1', desc=None, map_name="4x4",
               is_slippery=False, render_mode="rgb_array")
print("Action space:", env.action_space.n)
print("State space:", env.observation_space.n)

print(env.env.P[0])
state_size = 16
action_space = env.action_space.n
alpha = 0.05
gamma = .9
state_action_vals = np.random.randn(state_size, action_space)
policy = np.zeros(state_size, dtype=int)
episodes = 100
eps = 0.2
test_episodes = 5
test_every = 10
test_episode = []
rewards = []
outcomes = []


def select_action(s, eps):
    sample = np.random.uniform()
    if sample < eps:
        return env.action_space.sample()
    else:
        return np.argmax(state_action_vals[s])


for ep in range(episodes):
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4",
                   is_slippery=False, render_mode="rgb_array")
    s = env.reset()[0]
    a = select_action(s, eps)
    done = False
    outcomes.append(0)
    while not done:
        next_s, reward, done, trunc, _ = env.step(a)
        next_a = select_action(s, eps)

        action_value = state_action_vals[s, a]
        next_action_value = state_action_vals[next_s, next_a]
        delta = reward + gamma * next_action_value - action_value
        state_action_vals[s, a] += alpha * delta
        s, a = next_s, next_a
        if reward:
            outcomes[-1] = 1

    if ep % test_every == 0:
        total_rewards = 0
        for _ in range(test_episodes):
            done = False
            s = env.reset()[0]
            while not done:
                a = state_action_vals[s].argmax()
                s, reward, done, trunc, _ = env.step(a)
                total_rewards += reward
                if reward:
                    outcomes[-1] = 1
        rewards.append(total_rewards / test_episodes)
        test_episode.append(ep)

# N = 10


# def moving_average(a, n):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n


# means = np.array(outcomes)
# avg = moving_average(means, n=N)

# # Plot outcomes
# plt.figure(figsize=(12, 4))
# plt.title("Rewards vs Number of Episodes for restricted actions in Randomly Generated 4x4 Grids")
# plt.xlabel("Number of Episodes")
# plt.ylabel("Rewards")
# # Getting axes
# ax = plt.gca()
# plt.plot(avg)
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams.update({'font.size': 4})
# plt.show()

plt.figure(figsize=(12, 4))
plt.plot(rewards)

fig, ax = plt.subplots()
ax.plot(test_episode, rewards)
ax.set_title('Episodes vs average rewards')
ax.set_xlabel('Episode')
_ = ax.set_ylabel('Average reward')
plt.plot(ax)
plt.show()

from IPython.display import clear_output
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

environment = gym.make('FrozenLake-v1',
                       is_slippery=False, render_mode="ansi")
# q-table
q_matrix = np.zeros((environment.observation_space.n,
                     environment.action_space.n))

episodes = 1000000       # Total # of episodes
alpha = 0.05             # Learning rate
gamma = 0.9              # Discount factor

outcomes = []

for _ in range(episodes):
    done = False
    environment = gym.make('FrozenLake-v1', map_name="4x4",
                           is_slippery=False, render_mode="ansi")
    s = environment.reset()[0]

    outcomes.append(0)

    while not done:
        if np.max(q_matrix[s]) > 0:  # If there's action greater than 0
            a = np.argmax(q_matrix[s])
        else:
            a = environment.action_space.sample()  # Take random action
        new_s, r, done, trunc, info = environment.step(a)
        # Updating the q-table
        q_matrix[s, a] = q_matrix[s, a] + \
            alpha * (r + gamma * np.max(q_matrix[new_s]) - q_matrix[s, a])

        s = new_s
        if r:
            outcomes[-1] = 1

# print(q_matrix)

N = 100


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


means = np.array(outcomes)
av = moving_average(means, n=N)


# Plot outcomes
plt.figure(figsize=(12, 4))
plt.title("Rewards vs Number of Episodes in the same 4x4 Grids")
plt.xlabel("Number of Episodes")
plt.ylabel("Rewards")
# Getting axes
ax = plt.gca()
plt.plot(av)
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 4})
plt.show()

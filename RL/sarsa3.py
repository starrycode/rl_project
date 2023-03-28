from IPython.display import clear_output
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

environment = gym.make('FrozenLake-v1',
                       is_slippery=False, render_mode="ansi")
# Same grid 4x4
q_matrix2 = np.zeros((environment.observation_space.n,
                     environment.action_space.n))

epsilon = 0.2
episode_number = 300       # Total # of episodes
alpha = 0.05             # Learning rate
gamma = 0.9              # Discount factor


def choose_action(s):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = environment.action_space.sample()
    else:
        action = np.argmax(q_matrix2[s, :])
    return action


outcomes3 = []

for _ in range(episode_number):
    done = False
    environment = gym.make('FrozenLake-v1', map_name="4x4",
                           is_slippery=False, render_mode="ansi")
    s = environment.reset()[0]

    outcomes3.append(0)

    while not done:
        a = choose_action(s)
        new_s, r, done, trunc, info = environment.step(a)
        new_a = choose_action(new_s)

        action_value = q_matrix2[s, a]
        next_action_value = q_matrix2[new_s, new_a]
        delta = r + gamma * next_action_value - action_value
        q_matrix2[s, a] += alpha * delta

        # Updating the q-table
        # q_matrix2[s, a] = q_matrix2[s, a] + \
        #     alpha * (r + gamma * q_matrix2[new_s, new_a] - q_matrix2[s, a])

        s = new_s
        a = new_a
        if r:
            outcomes3[-1] = 1


# Visualizing the q_matrix-matrix
print(q_matrix2)

N = 10


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


means = np.array(outcomes3)
av = moving_average(means, n=N)

# Plot outcomes
plt.figure(figsize=(12, 4))
plt.title("Rewards vs Number of Episodes in the same 4x4 Grids")
plt.xlabel("Number of Episodes")
plt.ylabel("Rewards")
plt.plot(av)
# plt.plot(outcomes)

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 4})

plt.show()

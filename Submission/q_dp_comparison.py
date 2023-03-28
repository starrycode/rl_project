from IPython.display import clear_output
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

environment = gym.make('FrozenLake-v1',
                       is_slippery=False, render_mode="ansi")
# q-learning
q_matrix = np.zeros((environment.observation_space.n,
                     environment.action_space.n))

epi = 300                # Total # of episodes
alpha = 0.05             # Learning rate
gamma = 0.9              # Discount factor

outcomes = []

for _ in range(epi):
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

# DP:
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4",
               is_slippery=False, render_mode="rgb_array")

possible_states = env.observation_space.n
possible_actions = env.action_space.n
# state value table
V_values = np.zeros(env.observation_space.n)

# The optimal policy pi
optimal_pi = np.zeros(possible_states, dtype=int)

episodes = 1000
gamma = 0.9
delta_vals = []
outcomes2 = []
for i in range(episodes):
    delta = 0
    outcomes2.append(0)
    for states in range(possible_states):
        val = V_values[states]
        q_matrix = np.zeros(possible_states)
        for actions in range(possible_actions):
            for prob, next_state, reward, rand in env.P[states][actions]:
                q_matrix[actions] += prob * \
                    (reward + gamma * V_values[next_state])
        V_values[states] = np.max(q_matrix)
        delta = max(delta, np.abs(val - V_values[states]))
        optimal_pi[states] = np.argmax(q_matrix)
    delta_vals.append(delta)
    if reward:
        outcomes2[-1] = 1
    # check if the current and new state values
    # have converged and exit training if true
    if delta < .0000001:
        # print(f'State values converge after {i} episodes')
        break

# Testing the training results
episode_rewards = []
total_reward = 0
episodes = 300
for _ in range(episodes):
    state = env.reset()[0]
    ep_reward = 0
    end = False
    outcomes2.append(0)

    while not end:
        action = optimal_pi[state]
        state, reward, end, rand, rand = env.step(action)
        ep_reward += reward
    episode_rewards.append(ep_reward)
    total_reward += ep_reward
    if reward:
        outcomes2[-1] = 1

# SARSA:
environment = gym.make('FrozenLake-v1',
                       is_slippery=False, render_mode="ansi")
# Same grid 4x4
q_matrix2 = np.zeros((environment.observation_space.n,
                     environment.action_space.n))

epsilon = 0.3            # Epsilon value
episode_number = 300     # Total # of episodes
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
        if np.max(q_matrix2[s]) > 0:  # If there's action greater than 0
            a = choose_action(s)
        else:
            a = environment.action_space.sample()  # Take random action
        new_s, r, done, trunc, info = environment.step(a)
        new_a = choose_action(new_s)
        # Updating the q-table
        q_matrix2[s, a] = q_matrix2[s, a] + \
            alpha * (r + gamma * q_matrix2[new_s, new_a] - q_matrix2[s, a])

        s = new_s
        a = new_a
        if r:
            outcomes3[-1] = 1

N = 10


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


means = np.array(outcomes)
means2 = np.array(outcomes2)
means3 = np.array(outcomes3)
av = moving_average(means, n=N)
av2 = moving_average(means2, n=N)
av3 = moving_average(means3, n=N)

plt.figure(figsize=(12, 4))
plt.title("Rewards vs Number of Episodes in the same 4x4 Grids")
plt.xlabel("Number of Episodes")
plt.ylabel("Rewards")

# Plotting the two lines and specifying labels
plt.plot(av, label='Q-learning')
plt.plot(av2, label='Dynamic Programming')
plt.plot(av3, label='SARSA')

# Setting font size and dpi
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 4})

# Adding the legend
plt.legend(loc='lower right', fontsize='large')

# Displaying the plot
plt.show()

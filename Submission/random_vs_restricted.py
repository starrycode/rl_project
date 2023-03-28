from IPython.display import clear_output
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import random

environment = gym.make('FrozenLake-v1',
                       is_slippery=False, render_mode="ansi")
# Randomly generated
q_matrix = {}
# state = (TL, TR, BL, BR, L)
# action = (Left, Down, Right, Up)
# state_arr = ['S', 'F', 'F', 'G'] in TL, TR, BL, BR order (for 2x2)

grid_size = 4                # 4x4

episodes = 1000000           # Total # of episodes
alpha = 0.05                 # Learning rate
gamma = 0.9                  # Discount factor

outcomes = []

for _ in range(episodes):
    done = False
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=grid_size), map_name=None,
                   is_slippery=False, render_mode="ansi")
    env.reset()
    env.render()
    grid_desc = env.env.desc
    state_arr = []
    action_arr = [0, 0, 0, 0]
    outcomes.append(0)

    # Generating a current grid layout
    for i in range(grid_size * grid_size):
        state_desc = grid_desc.flat[i]
        if state_desc == b'S':
            state_arr.append('S')
        elif state_desc == b'F':
            state_arr.append('F')
        elif state_desc == b'H':
            state_arr.append('H')
        else:
            state_arr.append('G')

    # Putting position on the key
    state_arr.append(env.env.s)
    state_tuple = tuple(state_arr)

    # Adding the current to q_matrix
    if state_tuple not in q_matrix:
        q_matrix[state_tuple] = action_arr

    while not done:
        if max(q_matrix[state_tuple]) > 0:
            action_for_s = q_matrix[state_tuple]
            max_index = action_for_s.index(
                max(action_for_s))  # index == specific action
            # 0 = left, 1 = down, 2 = right, 3 = up
            a = max_index
        else:
            a = env.action_space.sample()  # Take random action

        #print(state_tuple, a)

        new_s, r, done, trunc, info = env.step(a)
        action_arr = [0, 0, 0, 0]

        new_state_arr = list(state_arr)  # not deep copy of list
        new_state_arr.pop()
        new_state_arr.append(env.env.s)  # Updating the tuple with new state
        new_state_tuple = tuple(new_state_arr)
        if new_state_tuple not in q_matrix:
            q_matrix[new_state_tuple] = action_arr
        # print(state_tuple)
        # print(new_state_tuple)

        # Updating the q-table
        q_matrix[state_tuple][a] = q_matrix[state_tuple][a] + alpha * \
            (r + gamma *
             max(q_matrix[new_state_tuple]) - q_matrix[state_tuple][a])
        state_tuple = new_state_tuple

        if r:
            outcomes[-1] = 1


# Restricted actions
q_matrix_res = {}

grid_size = 4

outcomes_res = []


def choose_action():
    if random.random() < 0.5:
        return 1
    else:
        return 2


for _ in range(episodes):
    done = False
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=grid_size), map_name=None,
                   is_slippery=False, render_mode="ansi")
    # env.action_space.n = 2
    env.reset()
    env.render()
    grid_desc = env.env.desc
    state_arr = []
    action_arr = [0, 0, 0, 0]
    outcomes_res.append(0)

    for i in range(grid_size * grid_size):
        state_desc = grid_desc.flat[i]
        if state_desc == b'S':
            state_arr.append('S')
        elif state_desc == b'F':
            state_arr.append('F')
        elif state_desc == b'H':
            state_arr.append('H')
        else:
            state_arr.append('G')

    state_arr.append(env.env.s)
    state_tuple = tuple(state_arr)

    if state_tuple not in q_matrix_res:
        q_matrix_res[state_tuple] = action_arr

    while not done:
        # If there's action greater than 0
        if max(q_matrix_res[state_tuple]) > 0:
            action_for_s = q_matrix_res[state_tuple]
            max_index = action_for_s.index(max(action_for_s))
            # 0 = left, 1 = down, 2 = right, 3 = up
            a = max_index
        else:
            a = choose_action()

        # print(state_tuple, a)

        new_s, r, done, trunc, info = env.step(a)
        # print(new_s, r, done, trunc, info)

        action_arr = [0, 0, 0, 0]

        new_state_arr = list(state_arr)  # not deep copy of list
        new_state_arr.pop()
        new_state_arr.append(env.env.s)  # Updating the tuple with new state
        new_state_tuple = tuple(new_state_arr)
        if new_state_tuple not in q_matrix_res:
            q_matrix_res[new_state_tuple] = action_arr
        # print(state_tuple)
        # print(new_state_tuple)

        # Updating the q-table
        q_matrix_res[state_tuple][a] = q_matrix_res[state_tuple][a] + alpha * \
            (r + gamma *
             max(q_matrix_res[new_state_tuple]) - q_matrix_res[state_tuple][a])
        state_tuple = new_state_tuple

        if r:
            outcomes_res[-1] = 1

N = 100


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


means = np.array(outcomes)
av = moving_average(means, n=N)

means_r = np.array(outcomes_res)
av_r = moving_average(means_r, n=N)

# Plot outcomes
plt.figure(figsize=(12, 4))
plt.title("Rewards vs Number of Episodes of Restricted & Non-restricted actions in Random 4x4 Grids")
plt.xlabel("Number of Episodes")
plt.ylabel("Rewards")

plt.plot(av, label='Non-restricted')
plt.plot(av_r, label='Restricted')

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 4})

plt.legend(loc='lower right', fontsize='large')

plt.show()

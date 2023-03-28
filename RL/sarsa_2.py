import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4",
               is_slippery=False, render_mode="rgb_array")
# Parameters
epsilon = 0.3
total_episodes = 10000
max_steps = 100
alpha = 0.05
gamma = 0.9
outcomes = []

# Initializing the q_matrix-vaue
q_matrix = np.zeros((env.observation_space.n, env.action_space.n))

# Function to choose the next action with episolon greedy


def choose_action(s):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_matrix[s, :])
    return action


# Starting the SARSA learning
for episode in range(total_episodes):
    t = 0
    s = env.reset()[0]
    a = choose_action(s)
    outcomes.append(0)
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4",
                   is_slippery=False, render_mode="rgb_array")
    done = False

    while not done:
        # Visualizing the training

        env.reset()
        env.render()

        next_s, reward, done, trunc, info = env.step(a)

        next_a = choose_action(next_s)

        # Learning the q_matrix-value
        q_matrix[s, a] = q_matrix[s, a] + alpha * \
            (reward + gamma * q_matrix[next_s,
             next_a] - q_matrix[s, a])

        s = next_s
        a = next_a

        # Updating the respective vaLues
        t += 1
        #reward += 1
        # if reward:
        #     outcomes[-1] = 1
        outcomes[-1] = reward

        # If at the end of learning process
        # if done:
        #     break


# Evaluating the performance
print("Performace : ", reward/total_episodes)

# Visualizing the q_matrix-matrix
print(q_matrix)

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
plt.plot(av)
# plt.plot(outcomes)

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 4})

plt.show()

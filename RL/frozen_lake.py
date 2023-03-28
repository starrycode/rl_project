import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random

episodes = 1000
alpha = 0.05
gamma = 0.9

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4",
               is_slippery=False, render_mode="human")
env.reset()
# action = env.action_space.sample()
# env.step(action)
# env.render()

# Our table has the following dimensions:
# (rows x columns) = (states x actions) = (16 x 4)
# qtable = np.zeros((16, 4))

# Alternatively, the gym library can also directly g
# give us the number of states and actions using
# "env.observation_space.n" and "env.action_space.n"
nb_states = env.observation_space.n  # = 16
nb_actions = env.action_space.n      # = 4
qtable = np.zeros((nb_states, nb_actions))

# Let's see how it looks
print('Q-table =')
print(qtable)

# random.choice(["L", "R", "U", "D"])
action = env.action_space.sample()

#  LEFT = 0
#  DOWN = 1
#  RIGHT = 2
#  UP = 3

# 2. Implement this action and move the agent in the desired direction
# (8, 0.0, False, False, {'prob': 1.0})
# next_state, reward, terminated, truncated , info = env.step(action)
new_state, reward, done, trunc, info = env.step(action)
print(env.step(action))

# Display the results (reward and map)
env.render()
print(f'Reward = {reward}')

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})

# We re-initialize the Q-table
qtable = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
episodes = 1000        # Total number of episodes
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor

# List of outcomes to plot
outcomes = []

print('Q-table before training:')
print(qtable)

# Training
for _ in range(episodes):
    state = env.reset()
    done = False

    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        # Choose the action with the highest value in the current state
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])

        # If there's no best action (only zeros), take a random one
        else:
            action = env.action_space.sample()

        # Implement this action and move the agent in the desired direction
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + \
            alpha * (reward + gamma *
                     np.max(qtable[new_state]) - qtable[state, action])

        # Update our current state
        state = new_state

        # If we have a reward, it means that our outcome is a success
        if reward:
            outcomes[-1] = "Success"

print()
print('===========================================')
print('Q-table after training:')
print(qtable)

# Plot outcomes
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()

episodes = 100
nb_success = 0


# Evaluation
for _ in range(100):
    state = env.reset()[0]
    done = False

    # Until the agent gets stuck or reaches the goal, keep training it
    while not done:
        # Choose the action with the highest value in the current state
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])

        # If there's no best action (only zeros), take a random one
        else:
            action = env.action_space.sample()

        # Implement this action and move the agent in the desired direction
        new_state, reward, done, info = env.step(action)

        # Update our current state
        state = new_state

        # When we get a reward, it means we solved the game
        nb_success += reward

# Let's check our success rate!
print(f"Success rate = {nb_success/episodes*100}%")

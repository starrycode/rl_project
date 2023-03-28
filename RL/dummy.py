import gymnasium as gym
import random
import numpy as np

environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode='ansi')
environment.reset()

action = environment.action_space.sample()
# print(environment.step(action))
print(environment.render())

# print(dir(environment.reset()))
print(environment.reset())
# Our table has the following dimensions:
# (rows x columns) = (states x actions) = (16 x 4)
qtable = np.zeros((16, 4))

# Alternatively, the gym library can also directly g
# give us the number of states and actions using
# "env.observation_space.n" and "env.action_space.n"
nb_states = environment.observation_space.n  # = 16
nb_actions = environment.action_space.n      # = 4
qtable = np.zeros((nb_states, nb_actions))

# Let's see how it looks
print('Q-table =')
print(qtable)

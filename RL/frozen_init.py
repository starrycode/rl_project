import gymnasium as gym
import matplotlib.pyplot as plt
import random
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="rgb_array")
# state = env.reset()
# action = env.action_space.sample()
# env.step(action)
# rgb_array = env.render()
# plt.imshow(rgb_array)
# plt.show()


env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), map_name=None,
               is_slippery=False, render_mode="human")

state = env.reset()
env.seed(random.randint(0, 8))
action = env.action_space.sample()
env.step(action)
env.render()

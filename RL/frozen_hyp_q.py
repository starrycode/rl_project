import gymnasium as gym
import numpy as np

# Hypothesis:
# The size of the Q-table affects the performance of the Q-learning algorithm

environment = gym.make('FrozenLake-v1', desc=None, map_name="4x4",
                       is_slippery=False, render_mode="rgb_array")

# TODO: Change the num of episodes later
episodes = 10        # Total # of episodes
alpha = 0.05           # Learning rate
gamma = 0.9            # Discount factor


def q_learning(environment, size):
    # q-table
    q_matrix = np.zeros((environment.observation_space.n,
                        environment.action_space.n))

    outcomes = []

    for e in range(episodes):
        done = False
        print('env:', environment)
        s = environment.reset()[0]
        print('e', e)
        print('s', s)

        outcomes.append("Failure")

        while not done:
            if np.max(q_matrix[s]) > 0:  # If there's action greater than 0
                a = np.argmax(q_matrix[s])
                print('a-if:', a)
            else:
                a = environment.action_space.sample()  # Take random action
                print('a-else:', a)
            new_s, r, done, trunc, info = environment.step(a)
            print('r?', r)
            # Updating the q-table
            q_matrix[s, a] = q_matrix[s, a] + \
                alpha * (r + gamma * np.max(q_matrix[new_s]) - q_matrix[s, a])

            s = new_s
            if r:
                outcomes[-1] = "Success"
        # if (e + 1) % 100 == 0:
            #print("Episode", e + 1, ": Reward =", np.sum(r))
    return q_matrix


# Experiment with different Q-table sizes
sizes_of_q_matrix = [16, 32, 64, 128, 256]

for s in sizes_of_q_matrix:
    print("q-matrix size:", s)
    q_matrix = q_learning(environment, size=s)

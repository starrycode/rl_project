import gymnasium as gym
import numpy as np

# Set the environment to FrozenLake
env = gym.make('FrozenLake-v1')

# Define the Q-learning algorithm


def q_learning(env, num_episodes=10000, alpha=0.8, gamma=0.95, epsilon=0.1):
    # Initialize the Q-table
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # Run the Q-learning algorithm for num_episodes
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # Choose an action using an epsilon-greedy policy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            # Take the chosen action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)

            # Update the Q-table using the Q-learning update rule
            q_table[state, action] = (1 - alpha) * q_table[state, action] + \
                alpha * (reward + gamma * np.max(q_table[next_state, :]))

            # Update the current state
            state = next_state

        # Decrease epsilon over time to reduce exploration
        epsilon = max(epsilon * 0.99, 0.01)

    return q_table


# Run the Q-learning algorithm and print the optimal policy
q_table = q_learning(env)

optimal_policy = np.argmax(q_table, axis=1)
print("Optimal policy:", optimal_policy)

import numpy as np

# Define the training data
X = np.array([[1, -3, 5], [1, -4, -2], [1, 2, 1], [1, 4, 3]])
y = np.array([1, 1, -1, -1])

# Define the initial weight and learning rate
w = np.zeros(3)
lr = 0.1

# Train the perceptron for a maximum of 3 iterations
for iter in range(3):
    i = 0  # for keeping track of y
    for x in X:
        if np.dot(w, x) > 0:
            a = 1
        else:
            a = -1
        for j in range(3):
            # Update the weights
            w[j] = w[j] + lr * (y[i] - a) * x[j]
        i += 1

    print(f"Weights after iter {iter+1}: {w}")

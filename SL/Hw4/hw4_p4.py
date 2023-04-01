import numpy as np

# Define the training data
X = np.array([[1, 1.2], [1, 2.8], [1, 2], [1, 0.9], [1, 5.1]])
y = np.array([3.2, 8.5, 4.7, 2.9, 11])
P = np.array([[1, 1.5], [1, 4.5]])

# Define the initial weight and learning rate
w = np.ones(2)
lr = 0.01

print("#4 (a)")
for iter in range(3):
    i = 0  # for keeping track of y
    delta_w = np.zeros(2)
    for x in X:
        o = np.dot(w, x)
        for j in range(2):
            delta_w[j] = delta_w[j] + lr * (y[i] - o) * x[j]
        i += 1
    for k in range(2):
        w[k] = w[k] + delta_w[k]

    print(f"Weights after iter {iter+1}: {w}")

print("#4 (b)")
print(np.dot(P, w))

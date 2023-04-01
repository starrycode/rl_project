from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

points = np.array([[1.2, 3.2], [2.8, 8.5], [2, 4.7], [0.9, 2.9], [5.1, 11]])
X = np.array([[1, 1.2], [1, 2.8], [1, 2], [1, 0.9], [1, 5.1]])
#y = np.array([3.2, 8.5, 4.7, 2.9, 11])

# X = np.array([[1.2, 3.2], [2.8, 8.5], [2, 4.7], [0.9, 2.9], [5.1, 11]])
y = np.arange(0, 6.1, 0.1)

distances = cdist(X, y.reshape(-1, 1))
nearest = np.argpartition(distances, 3, axis=0)[:3]
labels_equal = np.mean(X[nearest], axis=1)
weights = 1 / distances[nearest]
labels_weighted = np.sum(
    X[nearest] * weights, axis=1) / np.sum(weights, axis=1)
labels_weighted = np.array([[1.5, 3.49], [4.5, 10.43]])
plt.scatter(points[:, 0], points[:, 1], color='red', label='points')
# plt.plot(points)
plt.plot(y, labels_equal, label='3-NN Equal Weight')
plt.plot(y, labels_weighted, label='3-NN Distance Weighted')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# Set seed for reproducibility
np.random.seed(42)

# Linear data
X_linear = np.random.uniform(0, 5, (1000, 2))
y_linear = (X_linear[:, 1] > 0.5 * X_linear[:, 0] + 1).astype(int)

# Non-linear (circular) data
X_circular = np.random.uniform(0, 5, (1000, 2))
center = np.array([2, 2])
radius = 1.5
y_circular = (np.sqrt(np.sum((X_circular - center) ** 2, axis=1)) < radius).astype(int)

# Train single perceptron on linear data
linear_model = Perceptron(max_iter=1000)
linear_model.fit(X_circular, y_circular)

# Train 3-neuron MLP on circular data
mlp_model = MLPClassifier(hidden_layer_sizes=(3,), activation="relu", max_iter=20000)
mlp_model.fit(X_circular, y_circular)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Linear boundary
h = 0.05
xx, yy = np.meshgrid(np.arange(0, 5, h), np.arange(0, 5, h))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = linear_model.predict(grid).reshape(xx.shape)

ax1.contourf(xx, yy, Z, cmap="bwr", alpha=0.6)
# ax1.scatter(X_linear[:, 0], X_linear[:, 1], c=y_linear, cmap="bwr")
ax1.scatter(X_circular[:, 0], X_circular[:, 1], c=y_circular, cmap="bwr")
ax1.set_title("Single Perceptron (Linear Separation)")
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 5)

# Non-linear boundary
h = 0.05
xx, yy = np.meshgrid(np.arange(0, 5, h), np.arange(0, 5, h))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = mlp_model.predict(grid).reshape(xx.shape)

ax2.contourf(xx, yy, Z, cmap="bwr", alpha=0.6)
ax2.scatter(X_circular[:, 0], X_circular[:, 1], c=y_circular, cmap="bwr")
ax2.set_title("MLP with 3 Neurons (Non-Linear Separation)")
ax2.set_xlim(0, 5)
ax2.set_ylim(0, 5)

plt.show()

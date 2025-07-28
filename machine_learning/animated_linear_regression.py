import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 1 + 3 * X + 0.4 * np.random.randn(100, 1)

# Initialize parameters
m, b = 0, 0
learning_rate = 0.02
iterations = 100

# Store history
history = []

for _ in range(iterations):
    y_pred = m * X + b
    error = y_pred - y
    m_grad = (2 / len(X)) * np.sum(error * X)
    b_grad = (2 / len(X)) * np.sum(error)
    m -= learning_rate * m_grad
    b -= learning_rate * b_grad
    history.append((m, b))

# Animation
fig, ax = plt.subplots()
sc = ax.scatter(X, y, color="blue")
(line,) = ax.plot([], [], color="red")


def animate(i):
    m, b = history[i]
    y_line = m * X + b
    line.set_data(X, y_line)
    ax.set_title(f"Iteration {i + 1}: m={m:.2f}, b={b:.2f}")
    return (line,)


ani = animation.FuncAnimation(fig, animate, frames=len(history), interval=50)
plt.xlabel("Image Quality Score")
plt.ylabel("Fraud Risk Score")
plt.grid(True)
plt.show()

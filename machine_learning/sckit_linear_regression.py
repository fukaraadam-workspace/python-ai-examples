import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Simulated toy data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # true function: y = 4 + 3x + noise

# Fit model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", linewidth=2, label="Predicted Line")
plt.xlabel("Image Quality Score")
plt.ylabel("Fraud Risk Score")
plt.title("Linear Regression Model")
plt.legend()
plt.grid(True)
plt.show()

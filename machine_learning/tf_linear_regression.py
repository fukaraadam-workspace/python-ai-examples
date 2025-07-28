import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models, optimizers

# Simulated toy data
np.random.seed(0)
x = 2 * np.random.rand(100)
y = 5 + 2 * x + 4 * x**2 + np.random.randn(100)

# Combine x and x^2 into a single feature matrix
X_poly = np.hstack(((x**2)[:, np.newaxis], x[:, np.newaxis]))

# Define the TensorFlow model
model = models.Sequential(
    [
        layers.Input(shape=(2,)),  # Input layer with 2 features (x and x^2)
        layers.Dense(1),  # Output layer with 1 neuron for regression
    ]
)

# Compile the model
model.compile(optimizer=optimizers.SGD(learning_rate=0.01), loss="mse")

# Train the model
history = model.fit(X_poly, y, epochs=100, verbose=0)

# Predict values
y_pred = model.predict(X_poly)

# Sort x and corresponding predictions for smooth plotting
sorted_indices = np.argsort(x)  # Get indices that would sort x
x_sorted = x[sorted_indices]  # Sort x
y_pred_sorted = y_pred[sorted_indices]  # Sort predictions based on x

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(
    x_sorted, y_pred_sorted, color="red", linewidth=2, label="Predicted Line"
)  # Use sorted x and y_pred
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Model (Keras)")
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# Simulated data
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = (X > 5).astype(int).ravel()  # Tampered if score > 5

# Fit model
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X_test, y_prob, color="green", label="Predicted Probability")
plt.axhline(0.5, color="red", linestyle="--", label="Decision Threshold")
plt.xlabel("Score")
plt.ylabel("Probability of Classification")
plt.title("Logistic Regression Classification")
plt.legend()
plt.grid(True)
plt.show()

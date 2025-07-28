import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Simulated dataset: Hours studied vs. Pass/Fail (binary outcome)
np.random.seed(42)
hours_studied = np.random.uniform(1, 10, 100)  # Hours between 1 and 10
pass_fail = (hours_studied + np.random.randn(100) * 0.5 > 5).astype(
    int
)  # Pass if hours + noise > 5

# Reshape the data for sklearn
X = hours_studied.reshape(-1, 1)  # Feature: Hours studied
y = pass_fail  # Target: Pass (1) or Fail (0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(hours_studied, pass_fail, color="blue", label="Actual Data")
x_range = np.linspace(1, 10, 300).reshape(
    -1, 1
)  # Generate a range of hours for smooth curve
y_prob = model.predict_proba(x_range)[:, 1]  # Probability of passing
plt.plot(x_range, y_prob, color="red", linewidth=2, label="Logistic Regression Curve")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression: Predicting Exam Results")
plt.legend()
plt.grid(True)
plt.show()

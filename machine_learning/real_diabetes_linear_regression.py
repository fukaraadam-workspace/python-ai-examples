import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load full dataset
X, y = load_diabetes(return_X_y=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Create subplots for side-by-side graphs
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, color="purple", alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
axes[0].set_xlabel("Actual Disease Progression")
axes[0].set_ylabel("Predicted Progression")
axes[0].set_title("Actual vs Predicted (10 Features)")

# Plot 2: Feature Importance
feature_names = load_diabetes().feature_names
coefficients = model.coef_
axes[1].barh(feature_names, coefficients, color="teal")
axes[1].set_xlabel("Coefficient Value")
axes[1].set_title("Feature Importance in Linear Regression")

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

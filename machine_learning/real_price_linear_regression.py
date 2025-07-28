import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the Boston Housing Dataset
boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data[["RM"]]  # Use the average number of rooms per dwelling as the feature
y = boston.target  # Median value of owner-occupied homes in $1000s

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Model Coefficients: {model.coef_[0]:.2f}")
print(f"Model Intercept: {model.intercept_:.2f}")

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Median House Price ($1000s)")
plt.title("Linear Regression: Boston Housing Dataset")
plt.legend()
plt.grid(True)
plt.show()

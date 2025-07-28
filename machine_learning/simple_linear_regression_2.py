import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, learning_rate=0.01, n_iter=1000):
        """
        Train the linear regression model using gradient descent.

        Parameters:
        X : np.ndarray
            Input features (2D array of shape [n_samples, n_features]).
        y : np.ndarray
            Target values (1D array of shape [n_samples]).
        learning_rate : float
            Learning rate for gradient descent.
        n_iter : int
            Number of iterations for gradient descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # region PLOT SETUP START
        # Sort the x values for a smooth curve
        sorted_indices = np.argsort(
            X[:, -1]
        )  # Sort by the last column (original x values)
        sorted_x = X[sorted_indices, -1]  # Sorted x values
        sorted_X = X[sorted_indices]  # Sorted feature matrix
        # Set up the plot
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        ax.scatter(
            X[:, -1], y, color="blue", label="Data"
        )  # Use the last column of X for plotting
        (line,) = ax.plot(
            sorted_x,
            self.predict(sorted_X),
            color="red",
            label="Linear Fit",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.set_title("Linear Regression Training\ny = 4x^2 + 2x + 5")
        ax.legend()
        # Add a text annotation for MSE
        mse_text = ax.text(
            0.05,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
        plt.waitforbuttonpress()
        # endregion PLOT SETUP END

        for i in range(n_iter):
            # Predict values
            y_pred = self.predict(X)

            # Compute gradients
            dw = -(2 / n_samples) * np.dot(X.T, (y - y_pred))
            db = -(2 / n_samples) * np.sum(y - y_pred)

            # Update weights and bias
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            # region UPDATE THE PLOT START
            # Update the plot
            # MSE = Σ(yi - pi)² / n
            mse = np.mean((y - y_pred) ** 2)
            mse_text.set_text(
                f"Iteration {i + 1}\nMSE = {mse:.4f}\nweights = {[f'{w:.4f}' for w in self.weights]}\nbias = {self.bias:.4f}"
            )
            line.set_ydata(self.predict(sorted_X))  # Update the line
            # endregion UPDATE THE PLOT END
            plt.pause(0.01)  # Pause to create an animation effect

        # region FINALIZE PLOT START
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the final plot displayed
        # endregion FINALIZE PLOT END

    def predict(self, X):
        """
        Predict target values for given input features.

        Parameters:
        X : np.ndarray
            Input features (2D array of shape [n_samples, n_features]).

        Returns:
        np.ndarray
            Predicted target values (1D array of shape [n_samples]).
        """
        return np.dot(X, self.weights) + self.bias


# Generate synthetic data
np.random.seed(42)
x = 2 * np.random.rand(100)  # 100 samples
y = 4 * x**2 + 2 * x + 5  # y = 4x^2 + 2x + 5
# y = y + 1 * np.random.randn(100)  # Add some noise

# Combine x^2 and x into a single feature matrix
X = np.hstack(((x**2)[:, np.newaxis], x[:, np.newaxis]))

# Train the model
model = LinearRegression()
model.fit(X, y, learning_rate=0.01, n_iter=10000)

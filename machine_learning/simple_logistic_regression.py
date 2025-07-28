import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, learning_rate=0.1, n_iter=500):
        """
        Train logistic regression via gradient descent.

        X: np.ndarray, shape [n_samples, n_features]
        y: np.ndarray of 0/1 labels, shape [n_samples]
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # region PLOT SETUP START
        sorted_idx = np.argsort(X[:, -1])  # sort by original x
        sorted_x = X[sorted_idx, -1]
        sorted_X = X[sorted_idx]
        plt.ion()
        fig, ax = plt.subplots()
        ax.scatter(X[:, -1], y, c=y, cmap="bwr", edgecolor="k", label="Data (0/1)")
        (line,) = ax.plot(
            sorted_x,
            sigmoid(sorted_X.dot(self.weights) + self.bias),
            color="green",
            label="Logistic Regression Curve",
        )
        ax.set_xlabel("Hours Studied")
        ax.set_ylabel("Probability of Passing")
        ax.set_title("Logistic Regression: Predicting Exam Results")
        ax.legend()
        loss_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
        plt.waitforbuttonpress()
        # endregion PLOT SETUP END

        # ––– GRADIENT DESCENT LOOP –––
        for i in range(n_iter):
            # 1) Predict probabilities
            linear_output = X.dot(self.weights) + self.bias
            y_prob = sigmoid(linear_output)

            # 2) Compute gradients of BCE loss
            #    ∂L/∂w = (1/n) Xᵀ (p - y)
            #    ∂L/∂b = (1/n) Σ (p - y)
            error = y_prob - y
            dw = (1 / n_samples) * X.T.dot(error)
            db = (1 / n_samples) * np.sum(error)

            # 3) Update params
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            # region UPDATE THE PLOT START
            y_prob_line = sigmoid(sorted_X.dot(self.weights) + self.bias)
            line.set_ydata(y_prob_line)
            # compute current loss
            eps = 1e-15
            loss = -np.mean(
                y * np.log(y_prob + eps) + (1 - y) * np.log(1 - y_prob + eps)
            )
            loss_text.set_text(
                f"Iter {i + 1}/{n_iter}\nLoss = {loss:.4f}\n"
                f"w = {[f'{w:.3f}' for w in self.weights]}\nb = {self.bias:.3f}"
            )
            # endregion UPDATE THE PLOT END

            plt.pause(0.01)

        # region FINALIZE PLOT START
        plt.ioff()
        plt.show()
        # endregion FINALIZE PLOT END

    def predict_proba(self, X):
        z = X.dot(self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# Generate synthetic data
np.random.seed(42)
hours_studied = np.random.uniform(1, 10, 100)  # Hours between 1 and 10
pass_fail = (hours_studied + np.random.randn(100) * 0.5 > 5).astype(int)
X = hours_studied.reshape(-1, 1)  # Feature: Hours studied
y = pass_fail  # Target: Pass (1) or Fail (0)

# Train the model
model = LogisticRegression()
model.fit(X, y, learning_rate=1, n_iter=1000)

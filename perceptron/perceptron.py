import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sample_generator import generate_curve_separated_2d_data


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, sample_list: NDArray, labels: NDArray):
        """
        Train the Perceptron model on the given dataset.

        Parameters
        ----------
        sample_list : NDArray[np.float64]
            A 2D NumPy array where each row represents a sample and each column represents a feature.
        labels : NDArray[np.int64]
            A 1D NumPy array containing the target labels (0 or 1) for each sample.

        Returns
        -------
        None
        """
        n_samples, n_features = sample_list.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training the perceptron
        for _ in range(self.n_iter):
            for i, sample_i in enumerate(sample_list):
                label_predicted = self.predict(sample_i)

                # Update weights and bias
                update = self.learning_rate * (labels[i] - label_predicted)
                self.weights += update * sample_i
                self.bias += update

    def activation_function(self, predicted_output: NDArray[np.float64]) -> int:
        return np.where(predicted_output >= 0, 1, 0)

    def predict(self, sample_or_list: NDArray) -> NDArray:
        linear_output = np.dot(sample_or_list, self.weights) + self.bias
        label_predicted = self.activation_function(linear_output)
        return label_predicted

    def get_border_values_of_last_feature(self, sample_or_list: NDArray) -> NDArray:
        """
        Calculate the decision boundary of the perceptron.

        Returns
        -------
        NDArray[np.float64]
            A 1D NumPy array containing the border y.
        """
        border_y = (
            -(np.dot(sample_or_list[..., :-1], self.weights[:-1]) + self.bias)
            / self.weights[-1]
        )
        return border_y


def plot_results(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    labels: NDArray[np.int64],
    line_y: NDArray[np.float64],
    line_desc: str,
    border_y: NDArray[np.float64],
):
    plt.scatter(x, y, c=labels, cmap="bwr", edgecolor="k")
    sorted_i = np.argsort(x)
    plt.plot(x[sorted_i], line_y[sorted_i], color="green", label=line_desc)
    plt.plot(
        x[sorted_i],
        border_y[sorted_i],
        color="red",
        label="Perceptron Decision Boundary",
    )
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Points Above and Below the Line")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    x, y, labels, line_y, line_desc = generate_curve_separated_2d_data(100, -1, 0.5)
    perceptron = Perceptron(learning_rate=0.01, n_iter=100)
    sample_list = np.stack((np.square(x), x, y), axis=-1)
    perceptron.fit(sample_list, labels)

    border_y = perceptron.get_border_values_of_last_feature(sample_list)
    plot_results(x, y, labels, line_y, line_desc, border_y)

import matplotlib.pyplot as plt
import numpy as np


def generate_curve_separated_2d_data(
    n_samples: int, a2: float | None = None, a1: float | None = None
):
    x = np.random.rand(n_samples)
    y = np.random.rand(n_samples)

    if a2 is None:
        a2 = round(np.random.rand(), 2)
    if a1 is None:
        a1 = round(np.random.rand(), 2)
    a0 = round(0.5 - (a2 * 0.25 + a1 * 0.5), 2)
    line_y = a2 * np.square(x) + a1 * x + a0
    line_desc = f"y = {a2}x^2 + {a1}x + {a0}"

    # If y > a2 * x^2 + a1 * x + a0, label is 1, else label is 0
    labels = (y > line_y).astype(int)

    return x, y, labels, line_y, line_desc


def plot_data_with_line(
    x: list,
    y: list,
    labels: list,
    line_y: list,
    line_desc: str,
):
    plt.scatter(x, y, c=labels, cmap="bwr", edgecolor="k")
    sorted_i = np.argsort(x)
    plt.plot(x[sorted_i], line_y[sorted_i], color="green", label=line_desc)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Points Above and Below the Line")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    x, y, labels, line_y, line_desc = generate_curve_separated_2d_data(200, 0)
    plot_data_with_line(x, y, labels, line_y, line_desc)

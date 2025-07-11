import numpy as np


def main() -> None:
    x = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]
    x2 = [0.1, 0.2, 0.3, 0.4, 0.5]
    y = [[0.2], [0.4], [0.6], [0.8], [1.0]]
    x = np.array(x)
    x2 = np.array(x2)
    print(x[..., :-1])
    print(x2[..., :-1])


if __name__ == "__main__":
    main()

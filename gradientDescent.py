import numpy as np
from typing import Tuple
from computeCost import compute_cost


def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iters: int) -> Tuple[np.ndarray, np.ndarray]:
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):

        # hypothesis
        h = np.dot(X, theta)

        errors = h - y

        # calculate gradient
        gradient = (alpha / m) * np.dot(X.T, errors)

        # update theta
        theta = theta - gradient

        # save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
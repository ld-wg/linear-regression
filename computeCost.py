import numpy as np


def compute_cost(X, y, theta):
    m = y.size

    # hypothesis h = X * theta
    h = np.dot(X, theta)

    squared_errors = (h - y) ** 2

    # J(theta) = (1/(2m)) * sum(squared_errors)
    cost = (1 / (2 * m)) * np.sum(squared_errors)

    return cost

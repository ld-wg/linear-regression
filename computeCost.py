import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.

    # Calculate the hypothesis h = X * theta
    h = np.dot(X, theta)

    # Calculate the squared errors
    squared_errors = (h - y) ** 2

    # Calculate the cost function J(theta) = (1/(2m)) * sum(squared_errors)
    cost = (1 / (2 * m)) * np.sum(squared_errors)

    # ==========================================================

    return cost

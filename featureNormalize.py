import numpy as np


def feature_normalize(X):
    n = X.shape[1]  # the number of features
    X_norm = X
    mu = np.zeros(n)
    sigma = np.zeros(n)

    # calculate mean for each feature (column)
    mu = np.mean(X, axis=0)

    # calculate standard deviation for each feature (column)
    # use ddof=1 to match Octave's std function
    sigma = np.std(X, axis=0, ddof=1)

    # normalize X by subtracting mean and dividing by standard deviation
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

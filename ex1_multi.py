import matplotlib.pyplot as plt
import numpy as np
from featureNormalize import *
from gradientDescent import *
from normalEqn import *
from plotData import *

plt.ion()

# ===================== Part 1: Feature Normalization =====================
print('Loading Data...')
data = np.loadtxt('ex1data2.txt', delimiter=',', dtype=np.int64)
X = data[:, 0:2]
y = data[:, 2]
m = y.size

# print out some data points
print('First 10 examples from the dataset: ')
for i in range(0, 10):
    print('x = {}, y = {}'.format(X[i], y[i]))

input('Program paused. Press ENTER to continue')

# scale features and set them to zero mean
print('Normalizing Features ...')

X, mu, sigma = feature_normalize(X)
X = np.c_[np.ones(m), X]  # add a column of ones to X

# ===================== Part 2: Gradient Descent =====================

# ===================== Your Code Here =====================
# Instructions : We have provided you with the following starter
#                code that runs gradient descent with a particular
#                learning rate (alpha).
#
#                Your task is to first make sure that your functions -
#                computeCost and gradientDescent already work with
#                this starter code and support multiple variables.
#
#                After that, try running gradient descent with
#                different values of alpha and see which one gives
#                you the best result.
#
#                Finally, you should complete the code at the end
#                to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print('Running gradient descent ...')

# choose some alpha value
alpha = 0.03
num_iters = 400

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)

# plot the convergence graph
plot_convergence(J_history)

# display gradient descent's result
print('Theta computed from gradient descent : \n{}'.format(theta))

# Estimate the price of a 1650 sq-ft, 3 br house
# ===================== Your Code Here =====================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

# first, normalize the features for the test house
# (using the same mu and sigma calculated earlier)
test_house = np.array([1650, 3])
test_house_norm = (test_house - mu) / sigma
test_house_norm = np.insert(test_house_norm, 0, 1)  # Add intercept term

price = np.dot(test_house_norm, theta)

# ==========================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) : {:0.3f}'.format(price))

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Normal Equations =====================

print('Solving with normal equations ...')

# ===================== Your Code Here =====================
# Instructions : The following code computes the closed form
#                solution for linear regression using the normal
#                equations. You should complete the code in
#                normalEqn.py
#
#                After doing so, you should complete this code
#                to predict the price of a 1650 sq-ft, 3 br house.
#

# load data
data = np.loadtxt('ex1data2.txt', delimiter=',', dtype=np.int64)
X = data[:, 0:2]
y = data[:, 2]
m = y.size

# add intercept term to X
X = np.c_[np.ones(m), X]

theta = normal_eqn(X, y)

# display normal equation's result
print('Theta computed from the normal equations : \n{}'.format(theta))

# Estimate the price of a 1650 sq-ft, 3 br house
# ===================== Your Code Here =====================

# for normal equations, use the original (unnormalized) features
# since theta was calculated using the original features
test_house = np.array([1650, 3])
test_house = np.insert(test_house, 0, 1)  # Add intercept term (no normalization)

price = np.dot(test_house, theta)

# ==========================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations) : {:0.3f}'.format(price))

input('ex1_multi Finished. Press ENTER to exit')

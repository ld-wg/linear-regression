import matplotlib.pyplot as plt
import numpy as np
from computeCost import compute_cost
from gradientDescent import gradient_descent
from plotData import plot_data, plot_linear_fit, plot_cost_surface, plot_cost_contour

# ===================== Part 1: Plotting =====================
print('Plotting Data...')
data = np.loadtxt('ex1data1.txt', delimiter=',', usecols=(0, 1))
X = data[:, 0]
y = data[:, 1]
m = y.size

plt.ion()
plt.figure(0)
plot_data(X, y)

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Gradient descent =====================
print('Running Gradient Descent...')

X = np.c_[np.ones(m), X]  # add a column of ones to X
theta = np.zeros(2)  # initialize fitting parameters

# some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
print('Initial cost : ' + str(compute_cost(X, y, theta)) + ' (This value should be about 32.07)')

theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent: ' + str(theta.reshape(2)))

# plot the linear fit
plt.figure(0)
plot_linear_fit(X[:, 1], y, theta)

input('Program paused. Press ENTER to continue')

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of {:0.3f} (This value should be about 4519.77)'.format(predict1*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of {:0.3f} (This value should be about 45342.45)'.format(predict2*10000))

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Visualizing J(theta0, theta1) =====================
print('Visualizing J(theta0, theta1) ...')

# create grid of theta values for cost surface visualization
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# create meshgrid for 3D surface plot
xs, ys = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.zeros(xs.shape)

# calculate cost for each theta combination
for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i][j] = compute_cost(X, y, t)

# transpose for correct orientation in plots
J_vals = np.transpose(J_vals)

# plot 3D surface
plot_cost_surface(theta0_vals, theta1_vals, J_vals, theta_optimal=theta)

# plot contour
plot_cost_contour(theta0_vals, theta1_vals, J_vals, theta_optimal=theta, log_scale=True)

input('ex1 Finished. Press ENTER to exit')

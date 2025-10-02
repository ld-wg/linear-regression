import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm


def plot_data(x, y):
    """Plot training data as scatter plot."""
    plt.scatter(x, y, marker='x', c='r', s=50)
    plt.grid(True, alpha=0.3)


def plot_linear_fit(x, y, theta):
    """Plot data points and linear regression fit."""
    plot_data(x, y)

    # Plot regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = theta[0] + theta[1] * x_line
    plt.plot(x_line, y_line, 'b-', linewidth=2)


def plot_cost_surface(theta0_vals, theta1_vals, J_vals):
    """Plot 3D surface of cost function J(theta0, theta1)."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(theta0_grid, theta1_grid, J_vals, alpha=0.8)

    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'J($\theta$)')


def plot_cost_contour(theta0_vals, theta1_vals, J_vals):
    """Plot contour plot of cost function J(theta0, theta1)."""
    plt.figure()

    theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)
    cs = plt.contour(theta0_grid, theta1_grid, J_vals, levels=20)

    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.grid(True, alpha=0.3)
    plt.colorbar(cs)


def plot_convergence(J_history):
    """Plot convergence of cost function over iterations."""
    plt.plot(range(len(J_history)), J_history, 'b-', linewidth=2)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.grid(True, alpha=0.3)

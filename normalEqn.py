import numpy as np


def normal_eqn(X, y):
    # normal equation:
    #    θ = (X^T X)^(-1) X^T y

    # cost function:
    #    J(θ) = (1/(2m)) Σᵢ(h_θ(x^(i)) - y^(i))²

    # partial derivative:
    #    ∂J/∂θⱼ = (1/m) Σᵢ(h_θ(x^(i)) - y^(i)) × xⱼ^(i)
    
    # set derivative to zero (minimum condition):
    #    ∂J/∂θⱼ = 0 for all j
    
    # substitute hypothesis h_θ(x^(i)) = θ^T x^(i):
    #    (1/m) Σᵢ(θ^T x^(i) - y^(i)) × xⱼ^(i) = 0
    
    # rearrange:
    #    Σᵢ θ^T x^(i) xⱼ^(i) = Σᵢ y^(i) xⱼ^(i)

    # matrix notation:
    #    X^T X θ = X^T y

    # (X^T X)^(-1) X^T y = 0


    # X^T * X
    X_T_X = np.dot(X.T, X)

    # (X^T * X)^(-1)
    X_T_X_inv = np.linalg.inv(X_T_X)

    # X^T * y
    X_T_y = np.dot(X.T, y)

    # theta = (X^T * X)^(-1) * X^T * y
    theta = np.dot(X_T_X_inv, X_T_y)

    return theta

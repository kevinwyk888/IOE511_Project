"""
function, gradient, and hessian of each of the problem in the problem sets
"""
import numpy as np

# for p1-p4
def Quad_func(x,Q,q):
    return 0.5*x.T@Q@x + q.T@x

def Quad_grad(x,Q,q):
    return Q@x + q

def Quad_Hess(x,Q,q):
    return Q

# for p5-p6
def Quartic_func(x,Q,sigma):
    return 0.5*x.T@x + (sigma/4)*(x.T@Q@x)*(x.T@Q@x)

def Quartic_grad(x,Q,sigma):
    return x + sigma*(x.T@Q@x)*(Q@x)

def Quartic_hess(x,Q,sigma):
    return np.eye(len(x)) + sigma * (2 * np.outer(Q @ x, Q @ x) + (x.T@Q@x) * Q)

# for p7-p8
def Rosenbrock_func(x):
    return np.sum((1 - x[:-1])**2 + 100 * (x[1:] - x[:-1]**2)**2)

def Rosenbrock_grad(x):
    g = np.zeros_like(x)
    n = len(x)

    for i in range(n - 1):
        g[i] += -2 * (1 - x[i]) - 400 * x[i] * (x[i+1] - x[i]**2)
        g[i+1] += 200 * (x[i+1] - x[i]**2)

    return g


def Rosenbrock_hess(x):
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n - 1):
        H[i, i] += 2 - 400 * (x[i+1] - 3 * x[i]**2)
        H[i, i+1] += -400 * x[i]
        H[i+1, i] += -400 * x[i]
        H[i+1, i+1] += 200

    return H

# for p9
def Datafit_func(x):
    y = np.array([1.5,2.25,2.625])
    f = 0
    for i in range(3):
        a = x[0]*(1-x[1]**(i+1))
        f += (y[i]-a)**2
    return f

def Datafit_grad(x):
    y = np.array([1.5, 2.25, 2.625])
    g = np.zeros(2)

    for i in range(3):
        xi = i + 1
        pred = x[0] * (1 - x[1]**xi)
        diff = pred - y[i]

        g[0] += 2 * diff * (1 - x[1]**xi)
        g[1] += 2 * diff * (-x[0] * xi * x[1]**(xi - 1))

    return g

def Datafit_hess(x):
    y = np.array([1.5, 2.25, 2.625])
    H = np.zeros((2, 2))

    for i in range(3):
        k = i + 1
        r = x[0] * (1 - x[1]**k) - y[i]

        dr_dx1 = 1 - x[1]**k
        dr_dx2 = -x[0] * k * x[1]**(k - 1)

        d2r_dx1dx1 = 0.0
        d2r_dx1dx2 = -k * x[1]**(k - 1)
        d2r_dx2dx2 = -x[0] * k * (k - 1) * x[1]**(k - 2) if k >= 2 else 0.0

        H[0, 0] += 2 * (dr_dx1 * dr_dx1 + r * d2r_dx1dx1)
        H[0, 1] += 2 * (dr_dx1 * dr_dx2 + r * d2r_dx1dx2)
        H[1, 0] += 2 * (dr_dx2 * dr_dx1 + r * d2r_dx1dx2)
        H[1, 1] += 2 * (dr_dx2 * dr_dx2 + r * d2r_dx2dx2)
    return H

# for p10-p11
def Exponential_func(x):
    term1 = (np.exp(x[0]) - 1) / (np.exp(x[0]) + 1)
    term2 = 0.1 * np.exp(-x[0])
    term3 = np.sum((x[1:] - 1)**4)
    return term1 + term2 + term3

def Exponential_grad(x):
    g = np.zeros_like(x)
    expx = np.exp(x[0])
    denom = (expx + 1)

    g[0] = (2 * expx / (denom**2)) - 0.1 * np.exp(-x[0])
    g[1:] = 4 * (x[1:] - 1)**3
    return g

def Exponential_hess(x):
    n = len(x)
    H = np.zeros((n, n))
    expx = np.exp(x[0])
    denom = (expx + 1)

    H[0, 0] = (2 * expx * (1 - expx) / (denom**3)) + 0.1 * np.exp(-x[0])
    for i in range(1, n):
        H[i, i] = 12 * (x[i] - 1)**2
    return H

# for p12
def Genhumps_func(x):
    f = 0
    for i in range(len(x) - 1):
        f += np.sin(2*x[i])**2 * np.sin(2*x[i+1])**2 + 0.05 * (x[i]**2 + x[i+1]**2)
    return f


def Genhumps_grad(x):
    n = len(x)
    g = np.zeros(n)

    for i in range(n - 1):
        s1 = np.sin(2*x[i])
        c1 = np.cos(2*x[i])
        s2 = np.sin(2*x[i+1])
        c2 = np.cos(2*x[i+1])
        g[i] += 2*s1*c1 * 2 * s2**2 + 0.1*x[i]
        g[i+1] += 2*s2*c2 * 2 * s1**2 + 0.1*x[i+1]
    return g


def Genhumps_hess(x):
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n - 1):
        s1 = np.sin(2*x[i])
        c1 = np.cos(2*x[i])
        s2 = np.sin(2*x[i+1])
        c2 = np.cos(2*x[i+1])

        # diagonal terms
        H[i, i] += 4*(c1**2 - s1**2) * s2**2 + 0.1
        H[i+1, i+1] += 4*(c2**2 - s2**2) * s1**2 + 0.1

        # off-diagonal
        H[i, i+1] += 8*s1*c1*s2*c2
        H[i+1, i] += 8*s1*c1*s2*c2

    return H
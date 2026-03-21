import numpy as np

import algorithms
import functions


def Solver(problem, method, options):
    """Function that runs a chosen algorithm on a chosen problem

    Inputs:
        problem, method, options (structs)
    Outputs:
        final iterate (x) and final function value (f)
    """
    # compute initial function/gradient/Hessian
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    H = np.eye(x.shape[0]) # initialize the inverse Hessian approximation
    norm_g = np.linalg.norm(g, ord=np.inf)
    threshold = options.term_tol * max(norm_g, 1.0)

    f_hist = [float(f)]
    alpha_hist = []

    # set initial iteration counter
    k = 0

    while True:
        if norm_g <= threshold:
            break
        if k >= int(options.max_iterations):
            break
        match method.name:
            case "BFGS QN backtrack":

                alpha, x_new, f_new, g_new, H_new = algorithms.BFGS_qn_bt(x, H, g, problem, options, alpha=1.0, rho=0.5)

            case "BFGS QN wolfe":
                alpha, x_new, f_new, g_new, H_new = algorithms.BFGS_qn_wolfe(x, H, g, problem, options, alpha=1, alpha_high=1000, alpha_low=0, c=0.5)
                
            case _: # need to combine with other algorithms
                raise ValueError("method is not implemented yet")

        # update new function values
        x = x_new
        f = f_new
        g = g_new
        H = H_new
        norm_g = np.linalg.norm(g, ord=np.inf)

        f_hist.append(float(f))
        alpha_hist.append(float(alpha))
        # increment iteration counter
        k = k + 1

    return x, f, f_hist, alpha_hist

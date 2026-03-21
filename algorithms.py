import numpy as np

#backtracking helper func
def backtracking_line_search(x, f, g, d, problem, method):
    """
    f(x + alpha d) <= f(x) + c * alpha * g^T d
    """
    alpha = method.options.get("alpha0", 1.0)          # initial step
    rho   = method.options.get("rho", 0.5)             # shrink factor in (0,1)
    c     = method.options.get("c", 1e-4)              # Armijo constant in (0,1)

    # sanity: need descent direction
    gd = float(g.T @ d)
    if gd >= 0:
        # not a descent direction; safest is alpha=0 or flip direction
        # Here we raise to catch issues early:
        raise ValueError(f"Direction is not descent: g^T d = {gd} (should be < 0).")

    # backtracking loop
    max_ls_iters = 100
    ls_count = 0
    while ls_count < max_ls_iters:
        x_trial = x + alpha * d
        f_trial = problem.compute_f(x_trial)
        if f_trial <= f + c * alpha * gd:
            return alpha, x_trial, f_trial
        alpha *= rho
        ls_count += 1
    return alpha, x + alpha * d, problem.compute_f(x + alpha * d)


# Wolfe helper func
def wolfe_line_search(x, d, c1, c2, g, problem, alpha=1.0, alpha_high=1000.0, alpha_low=0.0, c=0.5, tol=1e-12, max_ls_iters=100):
    '''
    Wolfe line search using interval shrinking.
    Armijo condition:
        f(x + alpha d) <= f(x) + c1 * alpha * g^T d
    Curvature condition:
        g1^T d >= c2 * g^T d
    '''
    f0 = problem.compute_f(x)
    gd = float(g.T @ d)

    # Wolfe line search assumes a descent direction
    if gd >= 0:
        raise ValueError("d is not a descent direction: g^T d must be < 0.")

    for _ in range(max_ls_iters):
        x_new = x + alpha * d
        f = problem.compute_f(x_new)

        # Armijo condition
        if f <= f0 + c1 * alpha * gd:
            G = problem.compute_g(x_new)
            if float(G.T @ d) >= c2 * gd:
                return alpha, x_new, f

        # update interval
        if f <= f0 + c1 * alpha * gd:
            alpha_low = alpha
        else:
            alpha_high = alpha

        if abs(alpha_high - alpha_low) < tol: # in case the interval becomes too small
            x_new = x + alpha_low * d
            return alpha_low, x_new, problem.compute_f(x_new)

        alpha = c * alpha_low + (1 - c) * alpha_high

    x_new = x + alpha * d
    return alpha, x_new, problem.compute_f(x_new)

#BFGS quasi-newton's wolfe method
def BFGS_qn_wolfe(x, H, g, problem, options, alpha=1, alpha_high=1000, alpha_low=0, c=0.5):
    n = len(x)
    I = np.eye(n)

    d = -H @ g  # search direction
    if float(g.T @ d) >= 0:
        d = -g

    alpha, x_new, f_new = wolfe_line_search(  # apply wolfe
        x=x,
        d=d,
        c1=options.c1_ls,
        c2=options.c2_ls,
        g=g,
        problem=problem,
        alpha=alpha,
        alpha_high=alpha_high,
        alpha_low=0.0,
        c=c
    )

    g_new = problem.compute_g(x_new)

    s = x_new - x
    y = g_new - g
    ys = float(y.T @ s)

    if s.T @ y <= 1e-12: 
        H_new = H.copy()   # skip the update
    else:
        rho_k = 1.0 / ys
        H_new = (I - rho_k * s @ y.T) @ H @ (I - rho_k * y @ s.T) + rho_k * s @ s.T

    return alpha, x_new, f_new, g_new, H_new

#BFGS quasi-newton's backtracking method
def BFGS_qn_bt(x, H, g, problem, options, alpha=1.0, rho=0.5):
    n = len(x)
    I = np.eye(n)

    d = -H @ g
    if float(g.T @ d) >= 0:
        d = -g

    alpha, x_new, f_new = backtracking_line_search(  # apply backtracking, which is the only different place with BFGS_qn_wolfe
        x=x,
        d=d,
        c1=options.c1_ls,
        g=g,
        problem=problem,
        alpha=alpha,
        rho=rho
    )

    g_new = problem.compute_g(x_new)

    s = x_new - x
    y = g_new - g
    ys = float(y.T @ s)

    if ys <= 1e-12:
        H_new = H.copy() # skip the update
    else:
        rho_k = 1.0 / ys
        H_new = (I - rho_k * s @ y.T) @ H @ (I - rho_k * y @ s.T) + rho_k * s @ s.T

    return alpha, x_new, f_new, g_new, H_new

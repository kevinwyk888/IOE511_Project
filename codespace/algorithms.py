"""IOE 511/MATH 562, University of Michigan.

Core algorithm implementations used by ``optSolver.py``.
"""

import numpy as np


def _get_value(source, name, default):
    """Read a field from a dict-like or attribute-based container."""
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    if hasattr(source, name):
        return getattr(source, name)
    return default


def _get_option(method, options, name, default):
    """Read an option, giving precedence to method.options over options."""
    method_options = _get_value(method, "options", None)
    value = _get_value(method_options, name, None)
    if value is not None:
        return value
    return _get_value(options, name, default)


def _symmetrize(matrix):
    """Keep Hessian-like matrices numerically symmetric."""
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _ensure_descent_direction(g, d):
    """Fallback to steepest descent if the proposed direction is not descent."""
    if float(np.dot(g, d)) < 0.0:
        return d
    return -np.asarray(g, dtype=float).copy()


def _apply_step(problem, x, d, alpha):
    """Evaluate function and gradient after taking a step."""
    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)
    return x_new, f_new, g_new


def _backtracking_line_search(problem, x, f, g, d, method, options):
    """Armijo backtracking line search."""
    alpha = float(_get_option(method, options, "alpha0", 1.0))
    tau = float(_get_option(method, options, "tau", 0.5))
    c1 = float(_get_option(method, options, "c1_ls", 1e-4))
    max_iters = int(_get_option(method, options, "max_ls_iterations", 100))

    gTd = float(np.dot(g, d))
    if gTd >= 0.0:
        raise ValueError("Backtracking line search requires a descent direction.")

    for _ in range(max_iters):
        x_new, f_new, g_new = _apply_step(problem, x, d, alpha)
        if f_new <= f + c1 * alpha * gTd:
            return alpha, x_new, f_new, g_new
        alpha *= tau

    x_new, f_new, g_new = _apply_step(problem, x, d, alpha)
    return alpha, x_new, f_new, g_new


def _zoom(problem, x, f0, g0_dot_d, d, alpha_lo, alpha_hi, f_lo, method, options):
    """Bisection-style zoom step for the Wolfe line search."""
    c1 = float(_get_option(method, options, "c1_ls", 1e-4))
    c2 = float(_get_option(method, options, "c2_ls", 0.9))
    tol = float(_get_option(method, options, "wolfe_zoom_tol", 1e-12))
    max_iters = int(_get_option(method, options, "max_ls_iterations", 100))

    for _ in range(max_iters):
        alpha = 0.5 * (alpha_lo + alpha_hi)
        x_new, f_new, g_new = _apply_step(problem, x, d, alpha)

        if (f_new > f0 + c1 * alpha * g0_dot_d) or (f_new >= f_lo):
            alpha_hi = alpha
            continue

        g_new_dot_d = float(np.dot(g_new, d))
        if g_new_dot_d >= c2 * g0_dot_d:
            return alpha, x_new, f_new, g_new

        if g_new_dot_d * (alpha_hi - alpha_lo) >= 0.0:
            alpha_hi = alpha_lo

        alpha_lo = alpha
        f_lo = f_new

        if abs(alpha_hi - alpha_lo) <= tol:
            return alpha, x_new, f_new, g_new

    alpha = alpha_lo
    return alpha, *_apply_step(problem, x, d, alpha)


def _wolfe_line_search(problem, x, f, g, d, method, options):
    """Weak Wolfe line search with a zoom phase."""
    alpha_prev = 0.0
    alpha = float(_get_option(method, options, "alpha0", 1.0))
    alpha_max = float(_get_option(method, options, "alpha_max", 100.0))
    c1 = float(_get_option(method, options, "c1_ls", 1e-4))
    c2 = float(_get_option(method, options, "c2_ls", 0.9))
    max_iters = int(_get_option(method, options, "max_ls_iterations", 100))

    gTd = float(np.dot(g, d))
    if gTd >= 0.0:
        raise ValueError("Wolfe line search requires a descent direction.")

    f_prev = f
    for iteration in range(max_iters):
        x_new, f_new, g_new = _apply_step(problem, x, d, alpha)

        if (f_new > f + c1 * alpha * gTd) or (iteration > 0 and f_new >= f_prev):
            return _zoom(problem, x, f, gTd, d, alpha_prev, alpha, f_prev, method, options)

        g_new_dot_d = float(np.dot(g_new, d))
        if g_new_dot_d >= c2 * gTd:
            return alpha, x_new, f_new, g_new

        if g_new_dot_d >= 0.0:
            return _zoom(problem, x, f, gTd, d, alpha_prev, alpha, f_prev, method, options)

        alpha_prev = alpha
        f_prev = f_new
        alpha = min(2.0 * alpha, alpha_max)

    return alpha, x_new, f_new, g_new


def _modified_hessian(H, method, options):
    """Shift the Hessian until it becomes positive definite."""
    H = _symmetrize(H)
    n = H.shape[0]
    identity = np.eye(n)
    min_shift = float(_get_option(method, options, "min_hessian_shift", 1e-6))
    max_attempts = int(_get_option(method, options, "max_modify_iterations", 20))

    try:
        np.linalg.cholesky(H)
        return H
    except np.linalg.LinAlgError:
        diagonal_min = float(np.min(np.diag(H)))
        shift = max(min_shift, min_shift - diagonal_min)

    for _ in range(max_attempts):
        H_trial = H + shift * identity
        try:
            np.linalg.cholesky(H_trial)
            return H_trial
        except np.linalg.LinAlgError:
            shift = max(2.0 * shift, min_shift)

    return H + shift * identity


def _solve_modified_newton_direction(problem, x, g, method, options):
    """Compute a modified Newton direction that is guaranteed to be descent."""
    H = _modified_hessian(problem.compute_H(x), method, options)
    d = -np.linalg.solve(H, g)
    return _ensure_descent_direction(g, d)


def _trust_region_tau(p, d, delta):
    """Find the boundary intersection p + tau d with ||p + tau d|| = delta."""
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(p, d))
    c = float(np.dot(p, p) - delta**2)
    discriminant = max(b * b - 4.0 * a * c, 0.0)
    tau = (-b + np.sqrt(discriminant)) / (2.0 * a)
    return tau


def _truncated_cg(H, g, delta, tol, max_iters):
    """Steihaug-Toint CG solver for the trust-region Newton subproblem."""
    p = np.zeros_like(g, dtype=float)
    r = np.asarray(g, dtype=float).copy()
    d = -r

    if np.linalg.norm(r) <= tol:
        return p, False

    for _ in range(max_iters):
        Hd = H @ d
        dHd = float(np.dot(d, Hd))

        if dHd <= 0.0:
            tau = _trust_region_tau(p, d, delta)
            return p + tau * d, True

        rr = float(np.dot(r, r))
        alpha = rr / dHd
        p_next = p + alpha * d

        if np.linalg.norm(p_next) >= delta:
            tau = _trust_region_tau(p, d, delta)
            return p + tau * d, True

        r_next = r + alpha * Hd
        if np.linalg.norm(r_next) <= tol:
            return p_next, False

        beta = float(np.dot(r_next, r_next)) / rr
        d = -r_next + beta * d
        p = p_next
        r = r_next

    return p, False


def _bfgs_update(Hk, s, y):
    """Inverse-Hessian BFGS update."""
    ys = float(np.dot(y, s))
    if ys <= 1e-12:
        return Hk.copy()

    identity = np.eye(Hk.shape[0])
    rho = 1.0 / ys
    V = identity - rho * np.outer(s, y)
    return V @ Hk @ V.T + rho * np.outer(s, s)


def _sr1_update(Bk, s, y):
    """SR1 update for the Hessian approximation Bk.

    B_{k+1} = B_k + (y - B_k s)(y - B_k s)^T / ((y - B_k s)^T s)
    Skip update when denominator is too small (standard safeguard).
    """
    r = y - Bk @ s
    denom = float(np.dot(r, s))
    if abs(denom) < 1e-8 * np.linalg.norm(s) * np.linalg.norm(r):
        return Bk.copy()
    return Bk + np.outer(r, r) / denom


def _dfp_update(Hk, s, y):
    """DFP update for the inverse Hessian approximation Hk.

    H_{k+1} = H_k - (H_k y y^T H_k) / (y^T H_k y) + (s s^T) / (y^T s)
    """
    ys = float(np.dot(y, s))
    if ys <= 1e-12:
        return Hk.copy()
    Hy = Hk @ y
    yHy = float(np.dot(y, Hy))
    if abs(yHy) <= 1e-12:
        return Hk.copy()
    return Hk - np.outer(Hy, Hy) / yHy + np.outer(s, s) / ys


# Algorithm 1: GradientDescent, with backtracking line search.
def GradientDescent(x, f, g, problem, method, options):
    """Take one GradientDescent step using Armijo backtracking."""
    d = -np.asarray(g, dtype=float)
    alpha, x_new, f_new, g_new = _backtracking_line_search(problem, x, f, g, d, method, options)
    return x_new, f_new, g_new, d, alpha


# Algorithm 2: GradientDescentW, with Wolfe line search.
def GradientDescentW(x, f, g, problem, method, options):
    """Take one GradientDescentW step using a Wolfe line search."""
    d = -np.asarray(g, dtype=float)
    alpha, x_new, f_new, g_new = _wolfe_line_search(problem, x, f, g, d, method, options)
    return x_new, f_new, g_new, d, alpha


# Algorithm 3: Newton, modified Newton with backtracking line search.
def Newton(x, f, g, problem, method, options):
    """Take one modified-Newton step using Armijo backtracking."""
    d = _solve_modified_newton_direction(problem, x, g, method, options)
    alpha, x_new, f_new, g_new = _backtracking_line_search(problem, x, f, g, d, method, options)
    return x_new, f_new, g_new, d, alpha


# Algorithm 4: NewtonW, modified Newton with Wolfe line search.
def NewtonW(x, f, g, problem, method, options):
    """Take one modified-Newton step using a Wolfe line search."""
    d = _solve_modified_newton_direction(problem, x, g, method, options)
    alpha, x_new, f_new, g_new = _wolfe_line_search(problem, x, f, g, d, method, options)
    return x_new, f_new, g_new, d, alpha


# Algorithm 5: TRNewtonCG, trust region Newton with CG subproblem solver.
def TRNewtonCG(x, f, g, problem, method, options, trust_radius):
    """Take one trust-region Newton step using Steihaug CG."""
    H = _symmetrize(problem.compute_H(x))
    cg_tol_scale = float(_get_option(method, options, "term_tol_CG", 1e-6))
    cg_tol = cg_tol_scale * max(1.0, np.linalg.norm(g))
    max_cg_iters = int(_get_option(method, options, "max_iterations_CG", len(x)))
    c1_tr = float(_get_option(method, options, "c1_tr", 0.25))
    c2_tr = float(_get_option(method, options, "c2_tr", 0.75))
    gamma_dec = float(_get_option(method, options, "gamma1_tr", 0.25))
    gamma_inc = float(_get_option(method, options, "gamma2_tr", 2.0))
    max_tr_radius = float(_get_option(method, options, "max_tr_radius", 1000.0))

    d, boundary_hit = _truncated_cg(H, g, trust_radius, cg_tol, max_cg_iters)
    predicted_reduction = -(float(np.dot(g, d)) + 0.5 * float(np.dot(d, H @ d)))

    if predicted_reduction <= 0.0:
        return x, f, g, np.zeros_like(g), np.nan, max(gamma_dec * trust_radius, 1e-12)

    x_trial = x + d
    f_trial = problem.compute_f(x_trial)
    actual_reduction = f - f_trial
    rho = actual_reduction / predicted_reduction

    trust_radius_new = trust_radius
    if rho < c1_tr:
        trust_radius_new = max(gamma_dec * trust_radius, 1e-12)
    elif rho > c2_tr and boundary_hit:
        trust_radius_new = min(gamma_inc * trust_radius, max_tr_radius)

    if rho >= c1_tr:
        g_trial = problem.compute_g(x_trial)
        return x_trial, f_trial, g_trial, d, np.nan, trust_radius_new

    return x, f, g, np.zeros_like(g), np.nan, trust_radius_new


# Algorithm 6: TRSR1CG, SR1 quasi-Newton with CG subproblem solver.
def TRSR1CG(x, f, g, Bk, problem, method, options, trust_radius):
    """Take one SR1 trust-region step using Steihaug CG.

    Uses the SR1 Hessian approximation Bk in place of the true Hessian.
    Returns updated Bk along with the usual outputs.
    """
    B = _symmetrize(Bk)
    cg_tol_scale = float(_get_option(method, options, "term_tol_CG", 1e-6))
    cg_tol = cg_tol_scale * max(1.0, np.linalg.norm(g))
    max_cg_iters = int(_get_option(method, options, "max_iterations_CG", len(x)))
    c1_tr = float(_get_option(method, options, "c1_tr", 0.25))
    c2_tr = float(_get_option(method, options, "c2_tr", 0.75))
    gamma_dec = float(_get_option(method, options, "gamma1_tr", 0.25))
    gamma_inc = float(_get_option(method, options, "gamma2_tr", 2.0))
    max_tr_radius = float(_get_option(method, options, "max_tr_radius", 1000.0))

    # Solve the trust-region subproblem with truncated CG
    d, boundary_hit = _truncated_cg(B, g, trust_radius, cg_tol, max_cg_iters)
    predicted_reduction = -(float(np.dot(g, d)) + 0.5 * float(np.dot(d, B @ d)))

    if predicted_reduction <= 0.0:
        return x, f, g, np.zeros_like(g), np.nan, max(gamma_dec * trust_radius, 1e-12), Bk

    x_trial = x + d
    f_trial = problem.compute_f(x_trial)
    actual_reduction = f - f_trial
    rho = actual_reduction / predicted_reduction

    # Update trust-region radius
    trust_radius_new = trust_radius
    if rho < c1_tr:
        trust_radius_new = max(gamma_dec * trust_radius, 1e-12)
    elif rho > c2_tr and boundary_hit:
        trust_radius_new = min(gamma_inc * trust_radius, max_tr_radius)

    # Accept or reject the step
    if rho >= c1_tr:
        g_trial = problem.compute_g(x_trial)
        s = x_trial - x
        y = g_trial - g
        Bk_new = _sr1_update(Bk, s, y)
        return x_trial, f_trial, g_trial, d, np.nan, trust_radius_new, Bk_new

    return x, f, g, np.zeros_like(g), np.nan, trust_radius_new, Bk


# Algorithm 7: BFGS, BFGS quasi-Newton with backtracking line search.
def BFGS(x, f, g, Hk, problem, method, options):
    """Take one BFGS step using Armijo backtracking."""
    d = _ensure_descent_direction(g, -Hk @ g)
    alpha, x_new, f_new, g_new = _backtracking_line_search(problem, x, f, g, d, method, options)
    s = x_new - x
    y = g_new - g
    Hk_new = _bfgs_update(Hk, s, y)
    return x_new, f_new, g_new, d, alpha, Hk_new


# Algorithm 8: BFGSW, BFGS quasi-Newton with Wolfe line search.
def BFGSW(x, f, g, Hk, problem, method, options):
    """Take one BFGSW step using a Wolfe line search."""
    d = _ensure_descent_direction(g, -Hk @ g)
    alpha, x_new, f_new, g_new = _wolfe_line_search(problem, x, f, g, d, method, options)
    s = x_new - x
    y = g_new - g
    Hk_new = _bfgs_update(Hk, s, y)
    return x_new, f_new, g_new, d, alpha, Hk_new


# Algorithm 9: DFP, DFP quasi-Newton with backtracking line search.
def DFP(x, f, g, Hk, problem, method, options):
    """Take one DFP step using Armijo backtracking."""
    d = _ensure_descent_direction(g, -Hk @ g)
    alpha, x_new, f_new, g_new = _backtracking_line_search(problem, x, f, g, d, method, options)
    s = x_new - x
    y = g_new - g
    Hk_new = _dfp_update(Hk, s, y)
    return x_new, f_new, g_new, d, alpha, Hk_new


# Algorithm 10: DFPW, DFP quasi-Newton with Wolfe line search.
def DFPW(x, f, g, Hk, problem, method, options):
    """Take one DFPW step using a Wolfe line search."""
    d = _ensure_descent_direction(g, -Hk @ g)
    alpha, x_new, f_new, g_new = _wolfe_line_search(problem, x, f, g, d, method, options)
    s = x_new - x
    y = g_new - g
    Hk_new = _dfp_update(Hk, s, y)
    return x_new, f_new, g_new, d, alpha, Hk_new


def backtracking(problem, x, f, g, d, options):
    """Compatibility wrapper used by older code paths."""
    alpha, _, _, _ = _backtracking_line_search(problem, x, f, g, d, method=None, options=options)
    return alpha


def GDStep(x, f, g, problem, method, options):
    """Compatibility wrapper for the old Gradient Descent step helper."""
    return GradientDescent(x, f, g, problem, method, options)


def NewtonStep(x, f, g, problem, method, options):
    """Compatibility wrapper for the old Newton step helper."""
    return Newton(x, f, g, problem, method, options)


def backtracking_line_search(x, f, g, d, problem, method=None, options=None):
    """Compatibility wrapper that returns alpha, trial point, trial value."""
    alpha, x_new, f_new, _ = _backtracking_line_search(problem, x, f, g, d, method, options)
    return alpha, x_new, f_new


def wolfe_line_search(
    x,
    d,
    c1,
    c2,
    g,
    problem,
    alpha=1.0,
    alpha_high=1000.0,
    alpha_low=0.0,
    c=0.5,
    tol=1e-12,
    max_ls_iters=100,
):
    """Compatibility wrapper that returns alpha, trial point, trial value."""
    del alpha_high, alpha_low, c, tol
    x = np.asarray(x, dtype=float)
    options = {
        "alpha0": alpha,
        "c1_ls": c1,
        "c2_ls": c2,
        "max_ls_iterations": max_ls_iters,
    }
    f = problem.compute_f(x)
    alpha_out, x_new, f_new, _ = _wolfe_line_search(problem, x, f, g, d, method=None, options=options)
    return alpha_out, x_new, f_new


def BFGS_qn_wolfe(x, H, g, problem, options, alpha=1.0, alpha_high=1000.0, alpha_low=0.0, c=0.5):
    """Compatibility wrapper for Algorithm 8."""
    del alpha_high, alpha_low, c
    method = {"options": {"alpha0": alpha}}
    f = problem.compute_f(x)
    x_new, f_new, g_new, _, alpha_out, H_new = BFGSW(x, f, g, H, problem, method, options)
    return alpha_out, x_new, f_new, g_new, H_new


def BFGS_qn_bt(x, H, g, problem, options, alpha=1.0, rho=0.5):
    """Compatibility wrapper for Algorithm 7."""
    method = {"options": {"alpha0": alpha, "tau": rho}}
    f = problem.compute_f(x)
    x_new, f_new, g_new, _, alpha_out, H_new = BFGS(x, f, g, H, problem, method, options)
    return alpha_out, x_new, f_new, g_new, H_new

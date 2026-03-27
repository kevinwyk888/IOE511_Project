import numpy as np

import algorithms


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


def _initial_trust_radius(method, options):
    """Pick the initial trust-region radius from common option names."""
    for name in ("initial_trust_radius", "trust_radius0", "delta0"):
        value = _get_option(method, options, name, None)
        if value is not None:
            return max(float(value), 1e-12)
    return 1.0


def Solver(problem, method, options):
    """Run one chosen optimization algorithm on one chosen problem."""
    if _get_value(problem, "x0", None) is None:
        raise ValueError("problem.x0 is required")

    method_name = _get_value(method, "name", None)
    if method_name is None:
        raise ValueError("method.name is required")

    x = np.asarray(problem.x0, dtype=float).copy()
    f = problem.compute_f(x)
    g = np.asarray(problem.compute_g(x), dtype=float)

    term_tol = float(_get_option(method, options, "term_tol", 1e-6))
    max_iterations = int(_get_option(method, options, "max_iterations", 1e3))
    norm_g = np.linalg.norm(g, ord=np.inf)
    threshold = term_tol * max(norm_g, 1.0)

    # BFGS/BFGSW/DFP/DFPW keep an inverse-Hessian approximation; other methods ignore it.
    Hk = np.eye(x.shape[0])

    # TRSR1CG keeps a Hessian approximation Bk updated via SR1; other methods ignore it.
    Bk = np.eye(x.shape[0])

    # TRNewtonCG/TRSR1CG keeps its own trust-region radius from one iteration to the next.
    trust_radius = _initial_trust_radius(method, options)

    f_hist = [float(f)]
    alpha_hist = []

    k = 0
    while True:
        if norm_g <= threshold:
            break
        if k >= max_iterations:
            break

        match method_name:
            # Algorithm 1: GradientDescent, with backtracking line search.
            case "GradientDescent":
                x_new, f_new, g_new, _, alpha = algorithms.GradientDescent(x, f, g, problem, method, options)

            # Algorithm 2: GradientDescentW, with Wolfe line search.
            case "GradientDescentW":
                x_new, f_new, g_new, _, alpha = algorithms.GradientDescentW(x, f, g, problem, method, options)

            # Algorithm 3: Newton, modified Newton with backtracking line search.
            case "Newton":
                x_new, f_new, g_new, _, alpha = algorithms.Newton(x, f, g, problem, method, options)

            # Algorithm 4: NewtonW, modified Newton with Wolfe line search.
            case "NewtonW":
                x_new, f_new, g_new, _, alpha = algorithms.NewtonW(x, f, g, problem, method, options)

            # Algorithm 5: TRNewtonCG, trust region Newton with CG subproblem solver.
            case "TRNewtonCG":
                x_new, f_new, g_new, _, alpha, trust_radius = algorithms.TRNewtonCG(
                    x,
                    f,
                    g,
                    problem,
                    method,
                    options,
                    trust_radius,
                )
                
            # Algorithm 6: TRSR1CG, SR1 quasi-Newton with CG subproblem solver.
            case "TRSR1CG":
                x_new, f_new, g_new, _, alpha, trust_radius, Bk = algorithms.TRSR1CG(
                    x,
                    f,
                    g,
                    Bk,
                    problem,
                    method,
                    options,
                    trust_radius,
                )
                
            # Algorithm 7: BFGS, BFGS quasi-Newton with backtracking line search.
            case "BFGS" | "BFGS QN backtrack":
                x_new, f_new, g_new, _, alpha, Hk = algorithms.BFGS(x, f, g, Hk, problem, method, options)

            # Algorithm 8: BFGSW, BFGS quasi-Newton with Wolfe line search.
            case "BFGSW" | "BFGS QN wolfe":
                x_new, f_new, g_new, _, alpha, Hk = algorithms.BFGSW(x, f, g, Hk, problem, method, options)

            # Algorithm 9: DFP, DFP quasi-Newton with backtracking line search.
            case "DFP":
                x_new, f_new, g_new, _, alpha, Hk = algorithms.DFP(x, f, g, Hk, problem, method, options)

            # Algorithm 10: DFPW, DFP quasi-Newton with Wolfe line search.
            case "DFPW":
                x_new, f_new, g_new, _, alpha, Hk = algorithms.DFPW(x, f, g, Hk, problem, method, options)

            case _:
                raise ValueError(f"method '{method_name}' is not implemented yet")

        x = np.asarray(x_new, dtype=float)
        f = f_new
        g = np.asarray(g_new, dtype=float)
        norm_g = np.linalg.norm(g, ord=np.inf)

        f_hist.append(float(f))
        alpha_hist.append(float(alpha))
        k += 1

    return x, f, f_hist, alpha_hist

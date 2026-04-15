"""
Disk-neighborhood sensitivity analysis for GD with Wolfe line search.

Scope (kept deliberately small so it can slot into the poster):
  * One method: GradientDescentW.
  * One fixed center (c1*, c2*) = GD's best average parameter pair from
    the existing aggregate CSV.
  * Method 1 normalization (from the notes): log-scale c1, linear c2,
    min-max re-normalized to [0,1]^2.
  * Monte Carlo sampling of N points uniformly inside the disk
        N_r = { (u, v) : (z1(u) - z1*)^2 + (z2(v) - z2*)^2 <= r^2 }
  * Sensitivity score per problem p:
        S_p = (1/N) * sum_k (J_k - J*_p) / J*_p
    where J is f_eval at the given (c1, c2), and J*_p is f_eval at the
    fixed center on problem p (NOT the per-problem best).

Outputs:
  sensitivity_disk_GD_detailed.csv    one row per (problem, sample)
  sensitivity_disk_GD_summary.csv     one row per problem
  sensitivity_disk_GD_scatter.png     visualization of the sampled disk
  sensitivity_disk_GD_bar.png         per-problem S bar chart
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import functions
import optSolver as _optSolver_module


def optSolver(problem, method, options):
    if hasattr(_optSolver_module, "optSolver"):
        return _optSolver_module.optSolver(problem, method, options)
    return _optSolver_module.Solver(problem, method, options)


class Problem:
    def __init__(self, name, x0, func, grad, hess):
        self.name = name
        self.x0 = np.asarray(x0, dtype=float)
        self._func = func
        self._grad = grad
        self._hess = hess
        self.f_eval = 0
        self.g_eval = 0
        self.h_eval = 0

    def compute_f(self, x):
        self.f_eval += 1
        return float(self._func(np.asarray(x, dtype=float)))

    def compute_g(self, x):
        self.g_eval += 1
        return np.asarray(self._grad(np.asarray(x, dtype=float)), dtype=float)

    def compute_H(self, x):
        self.h_eval += 1
        return np.asarray(self._hess(np.asarray(x, dtype=float)), dtype=float)


class Method:
    def __init__(self, name, **options):
        self.name = name
        self.options = options


class Options:
    def __init__(self, term_tol=1e-6, max_iterations=1000, **kwargs):
        self.term_tol = term_tol
        self.max_iterations = max_iterations
        for key, value in kwargs.items():
            setattr(self, key, value)


def make_spd(n, kappa, rng):
    A = rng.standard_normal((n, n))
    Q_orth, _ = np.linalg.qr(A)
    eigvals = np.geomspace(1.0, float(kappa), num=n)
    return Q_orth @ np.diag(eigvals) @ Q_orth.T


def build_problem_specs(seed=0):
    rng = np.random.default_rng(seed)
    specs = []

    for name, n, kappa in [
        ("P1_quad_10_10", 10, 10),
        ("P2_quad_10_1000", 10, 1000),
        ("P3_quad_1000_10", 1000, 10),
        ("P4_quad_1000_1000", 1000, 1000),
    ]:
        Q = make_spd(n, kappa, rng)
        q = rng.uniform(-10.0, 10.0, size=n)
        x0 = rng.uniform(-10.0, 10.0, size=n)
        specs.append({
            "name": name,
            "x0": x0,
            "func": lambda x, Q=Q, q=q: functions.Quad_func(x, Q, q),
            "grad": lambda x, Q=Q, q=q: functions.Quad_grad(x, Q, q),
            "hess": lambda x, Q=Q, q=q: functions.Quad_Hess(x, Q, q),
        })

    Q4 = np.array([
        [5.0, 1.0, 0.0, 0.5],
        [1.0, 4.0, 0.5, 0.0],
        [0.0, 0.5, 3.0, 0.0],
        [0.5, 0.0, 0.0, 2.0],
    ])
    x0_q = np.array([np.cos(70.0), np.sin(70.0), np.cos(70.0), np.sin(70.0)])
    specs.append({
        "name": "P5_quartic_1e-4",
        "x0": x0_q,
        "func": lambda x: functions.Quartic_func(x, Q4, 1e-4),
        "grad": lambda x: functions.Quartic_grad(x, Q4, 1e-4),
        "hess": lambda x: functions.Quartic_hess(x, Q4, 1e-4),
    })
    specs.append({
        "name": "P6_quartic_1e4",
        "x0": x0_q,
        "func": lambda x: functions.Quartic_func(x, Q4, 1e4),
        "grad": lambda x: functions.Quartic_grad(x, Q4, 1e4),
        "hess": lambda x: functions.Quartic_hess(x, Q4, 1e4),
    })

    specs.append({
        "name": "P7_rosenbrock_2",
        "x0": np.array([-1.2, 1.0]),
        "func": functions.Rosenbrock_func,
        "grad": functions.Rosenbrock_grad,
        "hess": functions.Rosenbrock_hess,
    })
    x0_r100 = np.ones(100); x0_r100[0] = -1.2
    specs.append({
        "name": "P8_rosenbrock_100",
        "x0": x0_r100,
        "func": functions.Rosenbrock_func,
        "grad": functions.Rosenbrock_grad,
        "hess": functions.Rosenbrock_hess,
    })

    specs.append({
        "name": "P9_datafit_2",
        "x0": np.array([1.0, 1.0]),
        "func": functions.Datafit_func,
        "grad": functions.Datafit_grad,
        "hess": functions.Datafit_hess,
    })

    x0_e10 = np.zeros(10); x0_e10[0] = 1.0
    specs.append({
        "name": "P10_exponential_10",
        "x0": x0_e10,
        "func": functions.Exponential_func,
        "grad": functions.Exponential_grad,
        "hess": functions.Exponential_hess,
    })
    x0_e100 = np.zeros(100); x0_e100[0] = 1.0
    specs.append({
        "name": "P11_exponential_100",
        "x0": x0_e100,
        "func": functions.Exponential_func,
        "grad": functions.Exponential_grad,
        "hess": functions.Exponential_hess,
    })

    specs.append({
        "name": "P12_genhumps_5",
        "x0": np.array([-506.2, 506.2, -506.2, 506.2, -506.2]),
        "func": functions.Genhumps_func,
        "grad": functions.Genhumps_grad,
        "hess": functions.Genhumps_hess,
    })
    return specs


# ---------------------------------------------------------------------------
# Fixed experiment settings
# ---------------------------------------------------------------------------
METHOD_NAME = "GradientDescentW"

# Wolfe-parameter theoretical domain (c1 in (0, 1/2), c2 in (c1, 1)).
# Using the theoretical range (not the notebook grid) puts the center
# (0.05, 0.20) strictly in the interior of [0,1]^2, so the Monte Carlo
# disk is not clipped by the box.
C1_MIN, C1_MAX = 1e-5, 0.5
C2_MIN, C2_MAX = 1e-3, 0.999
LOG_C1_MIN = math.log10(C1_MIN)
LOG_C1_MAX = math.log10(C1_MAX)

# Fixed center — GD's best average pair across all 12 problems
C1_STAR = 0.05
C2_STAR = 0.20

# Disk radius in normalized [0,1]^2 space and Monte Carlo sample size
RADIUS = 0.15
N_SAMPLES = 100
SEED = 7
METRIC = "f_eval"

# Convergence tolerance for filtering non-converged problems
TERM_TOL = 1e-6
CONVERGE_TOL = 1e-4  # relaxed: accept "nearly solved" problems
MAX_ITER = 1000

GLOBAL_OPTIONS = {
    "term_tol": TERM_TOL,
    "max_iterations": MAX_ITER,
    "c1_ls": C1_STAR,
    "c2_ls": C2_STAR,
}
METHOD_DEFAULTS = {"alpha0": 1.0, "c1_ls": C1_STAR, "c2_ls": C2_STAR}


def to_z(c1, c2):
    z1 = (math.log10(c1) - LOG_C1_MIN) / (LOG_C1_MAX - LOG_C1_MIN)
    z2 = (c2 - C2_MIN) / (C2_MAX - C2_MIN)
    return z1, z2


def from_z(z1, z2):
    z1 = min(max(z1, 0.0), 1.0)
    z2 = min(max(z2, 0.0), 1.0)
    c1 = 10 ** (LOG_C1_MIN + z1 * (LOG_C1_MAX - LOG_C1_MIN))
    c2 = C2_MIN + z2 * (C2_MAX - C2_MIN)
    return c1, c2


def sample_disk(z1_star, z2_star, radius, n_samples, rng):
    pts = []
    max_tries = n_samples * 30
    tries = 0
    while len(pts) < n_samples and tries < max_tries:
        tries += 1
        u1, u2 = rng.random(), rng.random()
        rho = radius * math.sqrt(u1)
        theta = 2.0 * math.pi * u2
        z1 = z1_star + rho * math.cos(theta)
        z2 = z2_star + rho * math.sin(theta)
        if not (0.0 <= z1 <= 1.0 and 0.0 <= z2 <= 1.0):
            continue
        c1, c2 = from_z(z1, z2)
        if not (c1 < c2):
            continue
        pts.append((c1, c2, z1, z2))
    return pts


def run_one(problem_spec, c1_ls, c2_ls):
    method_options = dict(METHOD_DEFAULTS)
    method_options["c1_ls"] = c1_ls
    method_options["c2_ls"] = c2_ls

    merged_global = dict(GLOBAL_OPTIONS)
    merged_global["c1_ls"] = c1_ls
    merged_global["c2_ls"] = c2_ls

    problem = Problem(
        name=problem_spec["name"],
        x0=problem_spec["x0"],
        func=problem_spec["func"],
        grad=problem_spec["grad"],
        hess=problem_spec["hess"],
    )
    method = Method(METHOD_NAME, **method_options)
    options = Options(**merged_global)

    t0 = time.perf_counter()
    status = "ok"
    f_hist = []
    x_final = None
    try:
        out = optSolver(problem, method, options)
        if len(out) == 4:
            x_final, _, f_hist, _ = out
        else:
            x_final, _, f_hist = out
    except Exception as exc:
        status = f"fail:{exc}"
    cpu_sec = time.perf_counter() - t0
    iterations = max(len(f_hist) - 1, 0)

    grad_inf = np.nan
    if status == "ok" and x_final is not None:
        try:
            grad_inf = float(np.linalg.norm(
                problem.compute_g(x_final), ord=np.inf))
        except Exception:
            grad_inf = np.nan

    converged = (
        status == "ok"
        and iterations < MAX_ITER
        and np.isfinite(grad_inf)
        and grad_inf < CONVERGE_TOL
    )
    return {
        "status": status,
        "iterations": int(iterations),
        "f_eval": int(problem.f_eval),
        "g_eval": int(problem.g_eval),
        "cpu_sec": float(cpu_sec),
        "grad_inf": grad_inf,
        "converged": bool(converged),
    }


def main():
    specs = build_problem_specs(seed=0)

    z1_star, z2_star = to_z(C1_STAR, C2_STAR)
    print(f"Center (c1*, c2*) = ({C1_STAR}, {C2_STAR})  "
          f"-> (z1*, z2*) = ({z1_star:.3f}, {z2_star:.3f})")
    print(f"Disk radius r = {RADIUS} in normalized [0,1]^2 space, "
          f"N = {N_SAMPLES} Monte Carlo samples")

    rng = np.random.default_rng(SEED)
    samples = sample_disk(z1_star, z2_star, RADIUS, N_SAMPLES, rng)
    print(f"Obtained {len(samples)} valid samples inside the disk "
          f"(Wolfe constraint c1 < c2 enforced).\n")

    # One J* per problem: f_eval at the fixed center
    print("Running center (c1*, c2*) on every problem to get J*_p ...")
    center_info = {}
    for spec in specs:
        res = run_one(spec, C1_STAR, C2_STAR)
        center_info[spec["name"]] = res
        flag = "converged" if res["converged"] else "NOT converged"
        print(f"  {spec['name']:24s}  J*={res[METRIC]:<7}  "
              f"iter={res['iterations']:<5}  "
              f"|g|_inf={res['grad_inf']:.2e}  [{flag}]")

    converged_problems = [p for p in specs if center_info[p['name']]['converged']]
    dropped = [p['name'] for p in specs if not center_info[p['name']]['converged']]
    print(f"\nProblems kept (center converged): {len(converged_problems)} / {len(specs)}")
    if dropped:
        print(f"Dropped (center hit max_iter or |g| above tol): {dropped}")

    detailed_rows = []
    summary_rows = []

    print("\nRunning Monte Carlo samples on converged problems only ...")
    for spec in converged_problems:
        prob_name = spec["name"]
        J_star = float(center_info[prob_name][METRIC])
        if not np.isfinite(J_star) or J_star <= 0:
            print(f"  skip {prob_name}: invalid J* = {J_star}")
            continue

        rel_devs = []
        t_start = time.perf_counter()
        for k, (c1, c2, z1, z2) in enumerate(samples):
            res = run_one(spec, c1, c2)
            J = float(res[METRIC]) if res["status"] == "ok" else np.nan
            rel = (J - J_star) / J_star if np.isfinite(J) else np.nan
            rel_devs.append(rel)
            detailed_rows.append({
                "problem": prob_name,
                "sample_id": k,
                "c1_ls": c1,
                "c2_ls": c2,
                "z1": z1,
                "z2": z2,
                "status": res["status"],
                "iterations": res["iterations"],
                "f_eval": res["f_eval"],
                "g_eval": res["g_eval"],
                "cpu_sec": res["cpu_sec"],
                "grad_inf": res["grad_inf"],
                "converged": res["converged"],
                "J": J,
                "J_star": J_star,
                "rel_dev": rel,
            })
        arr = np.array([v for v in rel_devs if np.isfinite(v)])
        S = float(arr.mean()) if arr.size else np.nan
        S_med = float(np.median(arr)) if arr.size else np.nan
        S_std = float(arr.std(ddof=1)) if arr.size > 1 else np.nan
        S_max = float(arr.max()) if arr.size else np.nan
        failed = int(len([v for v in rel_devs if not np.isfinite(v)]))

        summary_rows.append({
            "problem": prob_name,
            "c1_star": C1_STAR,
            "c2_star": C2_STAR,
            "J_star": J_star,
            "radius": RADIUS,
            "n_valid": int(arr.size),
            "n_failed": failed,
            "S_mean": S,
            "S_median": S_med,
            "S_std": S_std,
            "S_max": S_max,
        })
        elapsed = time.perf_counter() - t_start
        print(f"  {prob_name:24s}  S_mean={S: .3f}  S_med={S_med: .3f}  "
              f"S_std={S_std: .3f}  n_fail={failed}  ({elapsed:.1f}s)")

    detailed_out = HERE / "sensitivity_disk_GD_detailed.csv"
    summary_out = HERE / "sensitivity_disk_GD_summary.csv"
    pd.DataFrame(detailed_rows).to_csv(detailed_out, index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_out, index=False)
    print(f"\nSaved detailed samples to:  {detailed_out}")
    print(f"Saved per-problem summary to: {summary_out}")

    overall_S = summary_df["S_mean"].mean()
    overall_med = summary_df["S_median"].mean()
    print(f"\nOverall sensitivity for GD around (c1*={C1_STAR}, c2*={C2_STAR}):")
    print(f"  mean of per-problem S_mean   = {overall_S:.3f}")
    print(f"  mean of per-problem S_median = {overall_med:.3f}")

    # -----------------------------------------------------------------------
    # One poster-ready figure: per-converged-problem sensitivity bar chart
    # with a small inset showing the sampling disk.
    # -----------------------------------------------------------------------
    if summary_df.empty:
        print("No converged problems — nothing to plot.")
        return

    # University-of-Michigan palette used in the existing poster
    UMICH_BLUE = "#00274C"
    UMICH_MAIZE = "#FFCB05"
    ACCENT = "#1F4E79"

    plot_df = summary_df.sort_values("S_mean", ascending=False).reset_index(drop=True)
    short_names = {
        "P1_quad_10_10": "P1 quad\n10/10",
        "P2_quad_10_1000": "P2 quad\n10/1000",
        "P3_quad_1000_10": "P3 quad\n1000/10",
        "P4_quad_1000_1000": "P4 quad\n1000/1000",
        "P5_quartic_1e-4": "P5 quart\n1e-4",
        "P6_quartic_1e4": "P6 quart\n1e4",
        "P7_rosenbrock_2": "P7 rosen\n2",
        "P8_rosenbrock_100": "P8 rosen\n100",
        "P9_datafit_2": "P9 datafit",
        "P10_exponential_10": "P10 exp\n10",
        "P11_exponential_100": "P11 exp\n100",
        "P12_genhumps_5": "P12 gen\nhumps",
    }

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    fig.patch.set_facecolor("white")

    xs = np.arange(len(plot_df))
    bars = ax.bar(
        xs, plot_df["S_mean"].values,
        yerr=plot_df["S_std"].fillna(0.0).values,
        capsize=4, color=UMICH_BLUE, alpha=0.9,
        edgecolor="white", linewidth=1.0,
        error_kw=dict(ecolor="#3b4a61", lw=1.2),
    )
    ax.axhline(
        overall_S, color=UMICH_MAIZE, lw=2.2, linestyle="--",
        label=fr"mean over solved problems: $\overline{{S}}={overall_S:.3f}$",
        zorder=2,
    )
    for bar, val in zip(bars, plot_df["S_mean"].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=9, color=UMICH_BLUE,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [short_names.get(n, n) for n in plot_df["problem"].values],
        fontsize=9,
    )
    ax.set_ylabel(
        r"Sensitivity  $S=\frac{1}{N}\sum_{k=1}^{N}\frac{J_k-J^*}{J^*}$",
        fontsize=11, color="#223",
    )
    ax.set_title(
        "GD Wolfe-parameter local sensitivity  "
        rf"(center $(c_1^*, c_2^*)=({C1_STAR}, {C2_STAR})$, "
        rf"disk $r={RADIUS}$, $N={N_SAMPLES}$, $J=$ f-evals)",
        fontsize=11.5, color=UMICH_BLUE, pad=10, weight="semibold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)

    ymax = max(0.05, float(plot_df["S_mean"].max()) * 1.45)
    ax.set_ylim(min(0.0, float(plot_df["S_mean"].min()) * 1.5), ymax)
    ax.legend(loc="upper left", fontsize=9, frameon=False,
              bbox_to_anchor=(0.02, 0.98))

    # ---- inset: the sampling disk in normalized space -------------------
    inset = fig.add_axes([0.74, 0.48, 0.18, 0.36])
    zs = np.array([(s[2], s[3]) for s in samples])
    inset.scatter(
        zs[:, 0], zs[:, 1], s=10, color=ACCENT,
        alpha=0.7, edgecolor="white", linewidth=0.3,
    )
    theta = np.linspace(0, 2 * np.pi, 200)
    inset.plot(
        z1_star + RADIUS * np.cos(theta),
        z2_star + RADIUS * np.sin(theta),
        color=UMICH_MAIZE, lw=1.8,
    )
    inset.scatter(
        [z1_star], [z2_star], s=80, marker="*", color="#E74C3C",
        edgecolor="black", linewidth=0.5, zorder=5,
    )
    inset.set_xlim(-0.02, 1.02)
    inset.set_ylim(-0.02, 1.02)
    inset.set_aspect("equal")
    inset.set_xticks([0, 1]); inset.set_yticks([0, 1])
    inset.tick_params(labelsize=7)
    inset.set_xlabel(r"$z_1$", fontsize=8, labelpad=0)
    inset.set_ylabel(r"$z_2$", fontsize=8, labelpad=-2)
    inset.set_title("sampling disk", fontsize=8.5, color=UMICH_BLUE)
    for s in ("top", "right"):
        inset.spines[s].set_visible(False)

    plt.tight_layout()
    out_path = HERE / "sensitivity_disk_GD_poster.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved poster figure to: {out_path}")


if __name__ == "__main__":
    main()

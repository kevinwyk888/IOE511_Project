# IOE 511 / MATH 562 — Phase II Deliverables

## Contents

### 1. Software Package (`.py`)
- `algorithms.py` — step-direction implementations for all 10 methods
  (GradientDescent, GradientDescentW, Newton, NewtonW, TRNewtonCG, TRSR1CG,
  BFGS, BFGSW, DFP, DFPW).
- `functions.py` — all 12 test problems (Quadratic, Quartic, Rosenbrock,
  Data fitting, Exponential, Genhumps) with function, gradient, and Hessian.
- `optSolver.py` — main solver entrypoint `optSolver(problem, method, options)`
  that dispatches to the method-specific routines in `algorithms.py`.

### 2. `run_summary_table.ipynb`
Runs the full benchmark of 12 problems × 10 algorithms = 120 runs and
produces:
- `Table: Summary of Results.csv` — per-run iterations, function/gradient/
  Hessian evaluations, CPU time, final objective, and final gradient norm.
- Convergence profile plots (one per problem) saved to
  `algorithm_performance/`.
- CPU-time performance profile plot.
- Big-Question sensitivity sweep over Wolfe parameters `(c1_ls, c2_ls)`
  exported to `phase2_c1_c2_sensitivity.csv`.

### 3. `run_algorithm_of_choice.ipynb`
Runs the chosen algorithm — **NewtonW with optimal Wolfe parameters
`(c1_ls, c2_ls) = (0.05, 0.5)`** from the Big-Question study — on both
Rosenbrock problems (n = 2 and n = 100). Produces:
- Summary table of iterations, evaluations, CPU time, and final gradient.
- Convergence plot saved to `algorithm_performance/algorithm_of_choice_rosenbrock.png`.

## Running

From this directory:

```bash
jupyter notebook run_summary_table.ipynb
jupyter notebook run_algorithm_of_choice.ipynb
```

Both notebooks add the current working directory to `sys.path`, so the
`.py` modules in this folder are imported directly.

## Dependencies

- Python 3.10+
- numpy
- matplotlib
- pandas (optional, used for nicer tables in `run_summary_table.ipynb`)

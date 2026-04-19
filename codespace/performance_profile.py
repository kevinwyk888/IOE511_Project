"""Build performance profiles for the top Wolfe parameter settings.

This script reads the detailed sensitivity-study CSV, selects the top-k
parameter pairs with the lowest average function-evaluation count for each
algorithm, and draws performance profiles for ``cpu_sec`` and ``f_eval``.

The comparison is done *within* each algorithm: every ``(c1_ls, c2_ls)``
pair is treated as one solver configuration for that algorithm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "wolfe_parameter_sensitivity_detailed.csv"
DEFAULT_OUTPUT_DIR = ROOT / "performance_profiles"
METRICS = ("cpu_sec", "f_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the top-k Wolfe parameter pairs per algorithm by avg_f_eval "
            "and draw performance profiles for cpu_sec and f_eval."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Detailed CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for figures and summary CSV. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of parameter pairs to keep per algorithm. Default: 10",
    )
    parser.add_argument(
        "--tau-points",
        type=int,
        default=400,
        help="Number of tau samples used in each profile. Default: 400",
    )
    return parser.parse_args()


def load_detailed_results(csv_path: Path) -> pd.DataFrame:
    """Read the detailed sensitivity CSV and coerce numeric columns."""
    df = pd.read_csv(csv_path)

    required_columns = {
        "problem",
        "method",
        "c1_ls",
        "c2_ls",
        "status",
        "f_eval",
        "cpu_sec",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Input CSV is missing required columns: {missing_str}")

    for column in ("c1_ls", "c2_ls", "f_eval", "cpu_sec", "iterations", "g_eval", "h_eval"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["status"] = df["status"].astype(str)
    return df


def summarize_parameter_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Average f_eval / cpu / iterations across problems for each
    (method, c1, c2) triple so configurations can be ranked."""
    success = (df["status"] == "ok").astype(float)
    work = df.copy()
    work["success"] = success

    summary = (
        work.groupby(["method", "c1_ls", "c2_ls"], as_index=False)
        .agg(
            avg_f_eval=("f_eval", "mean"),
            avg_cpu_sec=("cpu_sec", "mean"),
            avg_iterations=("iterations", "mean"),
            success_rate=("success", "mean"),
            n_rows=("problem", "size"),
            n_problems=("problem", "nunique"),
        )
        .sort_values(
            ["method", "avg_f_eval", "avg_cpu_sec", "success_rate", "c1_ls", "c2_ls"],
            ascending=[True, True, True, False, True, True],
        )
    )
    return summary


def select_top_configs(summary: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Keep the best ``top_k`` configurations per method (already sorted by
    ``summarize_parameter_pairs``) and attach a within-method rank."""
    top_configs = (
        summary.groupby("method", group_keys=False)
        .head(top_k)
        .copy()
    )
    top_configs["rank_within_method"] = top_configs.groupby("method").cumcount() + 1
    return top_configs


def config_label(row: pd.Series) -> str:
    """Short legend label like ``#3  c1=0.05, c2=0.9``."""
    c1 = f"{row['c1_ls']:g}"
    c2 = f"{row['c2_ls']:g}"
    return f"#{int(row['rank_within_method'])}  c1={c1}, c2={c2}"


def valid_metric_mask(frame: pd.DataFrame, metric: str) -> pd.Series:
    """Runs counted as "solved": status == ok and the metric is a finite positive number."""
    values = pd.to_numeric(frame[metric], errors="coerce")
    return (frame["status"] == "ok") & np.isfinite(values) & (values > 0)


def build_ratio_table(method_rows: pd.DataFrame, top_configs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Build the performance-profile ratio table r_{p,s} = t_{p,s} / min_s t_{p,s}.

    Rows are problems, columns are the top-k configurations, and unsolved
    runs are assigned ``+inf`` so they never contribute to ``min``.
    """
    problems = sorted(method_rows["problem"].dropna().unique())
    config_rows = top_configs.sort_values("rank_within_method")

    ratio_table = pd.DataFrame(index=problems)

    # Fill the raw metric values into the (problem x configuration) table.
    for _, config in config_rows.iterrows():
        mask = (
            (method_rows["c1_ls"] == config["c1_ls"])
            & (method_rows["c2_ls"] == config["c2_ls"])
        )
        config_runs = method_rows.loc[mask, ["problem", "status", metric]].copy()
        config_runs = config_runs.rename(columns={metric: "metric_value"})
        config_runs["metric_value"] = pd.to_numeric(config_runs["metric_value"], errors="coerce")
        config_runs["valid"] = valid_metric_mask(config_runs.rename(columns={"metric_value": metric}), metric)

        # Default to +inf; only overwrite for problems the configuration actually solved.
        values = pd.Series(np.inf, index=problems, dtype=float)
        valid_runs = config_runs[config_runs["valid"]].drop_duplicates(subset=["problem"], keep="first")
        if not valid_runs.empty:
            values.loc[valid_runs["problem"]] = valid_runs["metric_value"].to_numpy(dtype=float)

        ratio_table[config_label(config)] = values

    # Normalize each row by its best configuration to get the ratio r_{p,s}.
    best_by_problem = ratio_table.min(axis=1)

    for problem in ratio_table.index:
        best = best_by_problem.loc[problem]
        if not np.isfinite(best):
            # No configuration solved this problem; every ratio stays +inf.
            ratio_table.loc[problem, :] = np.inf
            continue
        ratio_table.loc[problem, :] = ratio_table.loc[problem, :] / best

    return ratio_table


def make_tau_grid(ratio_table: pd.DataFrame, tau_points: int) -> np.ndarray:
    """Choose a tau grid that spans [1, max finite ratio]; use log spacing when
    the spread is wide enough to need it."""
    finite_ratios = ratio_table.to_numpy(dtype=float)
    finite_ratios = finite_ratios[np.isfinite(finite_ratios)]

    if finite_ratios.size == 0:
        return np.linspace(1.0, 2.0, tau_points)

    tau_max = float(np.max(finite_ratios))
    tau_max = max(tau_max, 1.05)

    if tau_max <= 1.2:
        return np.linspace(1.0, tau_max, tau_points)

    return np.geomspace(1.0, tau_max, tau_points)


def make_tau_ticks(tau_max: float) -> list[float]:
    """Create readable tau ticks without scientific-notation offset text."""
    candidates = [
        1.0,
        1.1,
        1.2,
        1.5,
        2.0,
        3.0,
        5.0,
        7.0,
        10.0,
        15.0,
        20.0,
        30.0,
        50.0,
        70.0,
        100.0,
        150.0,
        200.0,
        300.0,
        500.0,
        700.0,
        1000.0,
    ]
    ticks = [tick for tick in candidates if 1.0 <= tick <= tau_max * 1.001]
    if len(ticks) >= 2:
        return ticks

    return list(np.linspace(1.0, tau_max, 5))


def plot_performance_profile(
    ratio_table: pd.DataFrame,
    method_name: str,
    metric: str,
    output_path: Path,
    tau_points: int,
) -> None:
    """Draw rho_s(tau) = (1/|P|) * #{p : r_{p,s} <= tau} for every configuration."""
    tau_grid = make_tau_grid(ratio_table, tau_points=tau_points)
    total_problems = len(ratio_table.index)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    columns_in_rank_order = list(ratio_table.columns)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, ratio_table.shape[1]))
    color_by_column = dict(zip(columns_in_rank_order, colors))
    handle_by_column = {}

    # Plot from worst rank to best rank so lower-numbered labels stay visible
    # when curves overlap. Also give smaller ranks higher zorder explicitly.
    for reverse_idx, column in enumerate(reversed(columns_in_rank_order), start=1):
        ratios = ratio_table[column].to_numpy(dtype=float)
        # rho_s(tau): fraction of problems this configuration solves within factor tau of the best.
        profile = np.array([(ratios <= tau).sum() / total_problems for tau in tau_grid], dtype=float)
        rank_idx = columns_in_rank_order.index(column) + 1
        line, = ax.plot(
            tau_grid,
            profile,
            lw=2.0,
            color=color_by_column[column],
            label=column,
            zorder=100 - rank_idx,
        )
        handle_by_column[column] = line

    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(float(tau_grid[0]), float(tau_grid[-1]))
    ax.set_xlabel("tau")
    ax.set_ylabel("Fraction of problems")
    ax.set_title(
        f"{method_name}: performance profile ({metric})\n"
        "Top 10 parameter pairs ranked by avg_f_eval"
    )
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    if tau_grid[-1] > 1.2:
        ax.set_xscale("log")
        tau_ticks = make_tau_ticks(float(tau_grid[-1]))
        ax.xaxis.set_major_locator(mticker.FixedLocator(tau_ticks))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.xaxis.offsetText.set_visible(False)
    legend_handles = [handle_by_column[column] for column in columns_in_rank_order]
    ax.legend(legend_handles, columns_in_rank_order, loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(top_configs: pd.DataFrame, output_dir: Path) -> Path:
    """Persist the per-method top-k configuration table alongside the plots."""
    output_path = output_dir / "performance_profile_top10_configs.csv"
    ordered = top_configs.sort_values(["method", "rank_within_method"]).copy()
    ordered.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    # Pipeline: load CSV -> rank configurations per method -> for each (method,
    # metric) pair build the ratio table and save its performance profile plot.
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Detailed CSV not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_detailed_results(input_path)
    summary = summarize_parameter_pairs(df)
    top_configs = select_top_configs(summary, top_k=args.top_k)
    summary_path = write_summary(top_configs, output_dir)

    methods = sorted(top_configs["method"].unique())

    for method_name in methods:
        method_rows = df[df["method"] == method_name].copy()
        method_top = top_configs[top_configs["method"] == method_name].copy()

        for metric in METRICS:
            ratio_table = build_ratio_table(method_rows, method_top, metric)
            output_path = output_dir / f"performance_profile_{method_name}_{metric}.png"
            plot_performance_profile(
                ratio_table,
                method_name,
                metric,
                output_path,
                tau_points=args.tau_points,
            )
            print(f"Saved {metric} profile for {method_name} to: {output_path}")

    print(f"Saved top-configuration summary to: {summary_path}")


if __name__ == "__main__":
    main()

"""
Experiment 2: Simulation study of bootstrap-calibrated Wasserstein budgets.

Reproduces Table 2 and Figure 4 from paper Section 5.2.

Bootstrap calibration procedure (paper p. 23-25):
  1. Draw B=500 bootstrap resamples of size n from the observed sample.
  2. For each resample, compute the required budget: the smallest ε making the
     general-class worst-case ES exceed the out-of-bag validation ES.
  3. Set epsilon_hat as the 475th order statistic of the 500 required budgets.
  4. Evaluate general, symmetric, and symmetric-unimodal worst-case ES at epsilon_hat.

Outputs saved to results/:
  exp2_summary_table.csv   (Table 2 equivalent)
  exp2_coverage_fig4.png   (Figure 4 equivalent)
  exp2_raw_results.csv     (per-repetition raw data)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    MU, SIGMA, ES_ALPHA,
    N_OUTER_PAPER, N_OUTER_RUN,
    N_BOOTSTRAP,
    SAMPLE_SIZES, EXP2_US_GRID, RANDOM_SEED, SOLVER_TOL,
)
from data.generate_synthetic import SAMPLING_DISTRIBUTIONS, population_es
from src.bootstrap_calibration import bootstrap_calibrate, evaluate_all_classes


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Alias for N_CALIBRATION_ORDER (0-indexed)
_CALIB_ORDER = 474   # 475th order statistic, 0-based


def _one_repetition(
    rep_idx: int,
    dist_name: str,
    n_samp: int,
    alpha: float,
    mu: float,
    sigma: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """
    Run one outer simulation repetition.

    Returns a dict with calibration outcome and class-specific ES bounds.
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Each repetition gets a unique RNG stream
    rng = np.random.default_rng([seed, rep_idx])

    sampler = SAMPLING_DISTRIBUTIONS[dist_name]["sampler"]
    sample = sampler(n_samp, rng)

    # Bootstrap calibration
    eps_hat, calibrated = bootstrap_calibrate(
        sample=sample,
        alpha=alpha,
        mu=mu,
        sigma=sigma,
        n_bootstrap=n_bootstrap,
        calibration_order=_CALIB_ORDER,
        rng=rng,
    )

    result: dict[str, Any] = {
        "rep": rep_idx,
        "dist": dist_name,
        "n": n_samp,
        "calibrated": calibrated,
        "eps_hat": eps_hat if calibrated else np.nan,
        "general": None,
        "symmetric": None,
        "sym_unimodal": None,
        "coverage_general": 0,
        "coverage_symmetric": 0,
        "coverage_sym_unimodal": 0,
    }

    if not calibrated:
        return result

    # Post-calibration evaluation
    try:
        bounds = evaluate_all_classes(
            sample=sample,
            eps_hat=eps_hat,
            alpha=alpha,
            mu=mu,
            sigma=sigma,
            upper_half_grid=EXP2_US_GRID,
            solver_tol=SOLVER_TOL,
        )
    except Exception:
        return result

    result["general"] = bounds.get("general")
    result["symmetric"] = bounds.get("symmetric")
    result["sym_unimodal"] = bounds.get("sym_unimodal")

    return result


def run_simulation(
    dist_name: str,
    n_samp: int,
    n_outer: int,
    n_bootstrap: int,
    pop_es: float,
    alpha: float,
    mu: float,
    sigma: float,
    seed: int,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Run all outer repetitions for one (distribution, sample size) combination."""
    futures_map = {}
    rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for rep in range(n_outer):
            fut = executor.submit(
                _one_repetition, rep, dist_name, n_samp, alpha, mu, sigma,
                n_bootstrap, seed + rep,
            )
            futures_map[fut] = rep

        for fut in as_completed(futures_map):
            try:
                result = fut.result(timeout=120)
            except Exception as exc:
                rep = futures_map[fut]
                result = {
                    "rep": rep, "dist": dist_name, "n": n_samp,
                    "calibrated": False, "eps_hat": np.nan,
                    "general": None, "symmetric": None, "sym_unimodal": None,
                    "coverage_general": 0, "coverage_symmetric": 0,
                    "coverage_sym_unimodal": 0,
                }
                warnings.warn(f"Rep {rep} failed: {exc}")

            # Coverage: 1 if bound >= pop_es, else 0 (calibration failure → 0)
            for cls in ("general", "symmetric", "sym_unimodal"):
                v = result.get(cls)
                result[f"coverage_{cls}"] = int(
                    result["calibrated"] and v is not None and v >= pop_es - 1e-10
                )
            rows.append(result)

    return rows


def aggregate_metrics(rows: list[dict], pop_es: float) -> dict[str, float]:
    """Aggregate repetition-level results into Table 2 metrics."""
    n_reps = len(rows)
    calibrated = [r for r in rows if r["calibrated"]]
    n_cal = len(calibrated)

    cal_rate = n_cal / n_reps if n_reps > 0 else 0.0
    median_eps = float(np.nanmedian([r["eps_hat"] for r in calibrated])) if calibrated else np.nan

    # Feasibility: fraction with non-None bound
    def feasibility(cls: str) -> float:
        feas = sum(1 for r in calibrated if r[cls] is not None) / max(n_cal, 1)
        return feas

    # Coverage: fraction of ALL reps where bound >= pop_ES (paper counts cal failures as 0)
    def coverage(cls: str) -> float:
        return sum(r[f"coverage_{cls}"] for r in rows) / max(n_reps, 1)

    # Mean WC-ES over the common feasible subset (all three classes feasible)
    common_feasible = [
        r for r in calibrated
        if all(r[c] is not None for c in ("general", "symmetric", "sym_unimodal"))
    ]
    n_cf = len(common_feasible)

    mean_wc = {}
    for cls in ("general", "symmetric", "sym_unimodal"):
        if common_feasible:
            mean_wc[cls] = float(np.mean([r[cls] for r in common_feasible]))
        else:
            mean_wc[cls] = np.nan

    return {
        "cal_rate": cal_rate,
        "median_eps": median_eps,
        "feas_general": feasibility("general"),
        "feas_symmetric": feasibility("symmetric"),
        "feas_sym_unimodal": feasibility("sym_unimodal"),
        "coverage_general": coverage("general"),
        "coverage_symmetric": coverage("symmetric"),
        "coverage_sym_unimodal": coverage("sym_unimodal"),
        "mean_wc_general": mean_wc["general"],
        "mean_wc_symmetric": mean_wc["symmetric"],
        "mean_wc_sym_unimodal": mean_wc["sym_unimodal"],
        "n_common_feasible": n_cf,
        "n_reps": n_reps,
        "n_calibrated": n_cal,
    }


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    denom = 1.0 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def plot_coverage_figure(all_metrics: dict, n_outer: int, save_path: Path) -> None:
    """
    Plot coverage and mean WC-ES as a function of sample size (Figure 4 equivalent).
    """
    dist_names = list(SAMPLING_DISTRIBUTIONS.keys())
    n_sizes = SAMPLE_SIZES
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharey="row")

    for d_idx, dist_name in enumerate(dist_names):
        label = SAMPLING_DISTRIBUTIONS[dist_name]["label"]
        ax_cov = axes[0, d_idx]
        ax_mean = axes[1, d_idx]

        for cls, clr, ls, cls_label in [
            ("general", "#1f77b4", "-", "General"),
            ("symmetric", "#ff7f0e", "--", "Symmetric"),
            ("sym_unimodal", "#d62728", ":", "Sym. unimodal"),
        ]:
            coverages = []
            lo_bands, hi_bands = [], []
            means = []
            for n_s in n_sizes:
                m = all_metrics.get((dist_name, n_s), {})
                cov = m.get(f"coverage_{cls}", np.nan)
                coverages.append(cov)
                lo, hi = wilson_ci(cov, n_outer)
                lo_bands.append(lo); hi_bands.append(hi)
                means.append(m.get(f"mean_wc_{cls}", np.nan))

            ax_cov.plot(n_sizes, coverages, color=clr, linestyle=ls,
                        marker="o", ms=5, label=cls_label, linewidth=1.6)
            ax_cov.fill_between(n_sizes, lo_bands, hi_bands, color=clr, alpha=0.15)
            ax_mean.plot(n_sizes, means, color=clr, linestyle=ls,
                         marker="s", ms=5, label=cls_label, linewidth=1.6)

        ax_cov.axhline(0.95, color="gray", linestyle="--", linewidth=1, label="95%")
        ax_cov.set_title(label, fontsize=10)
        ax_cov.set_xscale("log")
        ax_cov.set_ylim(-0.05, 1.05)
        ax_cov.set_xlabel("n"); ax_cov.set_ylabel("Coverage" if d_idx == 0 else "")
        ax_cov.legend(fontsize=7)

        ax_mean.set_xscale("log")
        ax_mean.set_xlabel("n"); ax_mean.set_ylabel("Mean WC-ES" if d_idx == 0 else "")
        ax_mean.legend(fontsize=7)

    fig.suptitle("Coverage and mean worst-case ES by sample size (Fig. 4 equivalent)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: Bootstrap simulation study")
    parser.add_argument("--n-outer", type=int, default=N_OUTER_RUN,
                        help=f"Number of outer repetitions (paper: {N_OUTER_PAPER})")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Number of bootstrap resamples")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker count")
    parser.add_argument("--full", action="store_true",
                        help=f"Run the full paper configuration (n_outer={N_OUTER_PAPER})")
    args = parser.parse_args()

    n_outer = N_OUTER_PAPER if args.full else args.n_outer
    n_bootstrap = N_BOOTSTRAP
    alpha = ES_ALPHA

    print(f"Exp 2: n_outer={n_outer}, n_bootstrap={n_bootstrap}, alpha={alpha}")
    print(f"  Distributions: {list(SAMPLING_DISTRIBUTIONS.keys())}")
    print(f"  Sample sizes: {SAMPLE_SIZES}")

    # Pre-compute population ES values
    pop_es_values: dict[str, float] = {}
    for dist_name in SAMPLING_DISTRIBUTIONS:
        pe = population_es(dist_name, alpha)
        pop_es_values[dist_name] = pe
        print(f"  Population ES_{alpha}({dist_name}) = {pe:.4f}")

    all_raw: list[dict] = []
    all_metrics: dict = {}

    t0 = time.time()
    for dist_name in SAMPLING_DISTRIBUTIONS:
        for n_samp in SAMPLE_SIZES:
            print(f"\n  Running {dist_name}, n={n_samp} ...", end=" ", flush=True)
            seed = RANDOM_SEED + abs(hash(dist_name)) % 10000 + n_samp
            t1 = time.time()

            rows = run_simulation(
                dist_name=dist_name,
                n_samp=n_samp,
                n_outer=n_outer,
                n_bootstrap=n_bootstrap,
                pop_es=pop_es_values[dist_name],
                alpha=alpha,
                mu=MU,
                sigma=SIGMA,
                seed=seed,
                max_workers=args.workers,
            )
            all_raw.extend(rows)
            m = aggregate_metrics(rows, pop_es_values[dist_name])
            all_metrics[(dist_name, n_samp)] = m
            t2 = time.time()

            print(
                f"done in {t2-t1:.0f}s | "
                f"cal={m['cal_rate']:.2%} | "
                f"feas_gen={m['feas_general']:.2%} | "
                f"feas_sym={m['feas_symmetric']:.2%} | "
                f"cov_gen={m['coverage_general']:.2%} | "
                f"cov_sym={m['coverage_symmetric']:.2%}"
            )

    total_time = time.time() - t0
    print(f"\nTotal simulation time: {total_time:.0f}s")

    # Save raw results
    raw_df = pd.DataFrame(all_raw)
    raw_df.to_csv(RESULTS_DIR / "exp2_raw_results.csv", index=False)
    print(f"  Saved: exp2_raw_results.csv ({len(raw_df)} rows)")

    # Build summary table (Table 2 equivalent)
    table_rows = []
    for dist_name in SAMPLING_DISTRIBUTIONS:
        for n_samp in SAMPLE_SIZES:
            m = all_metrics.get((dist_name, n_samp), {})
            row = {
                "Distribution": SAMPLING_DISTRIBUTIONS[dist_name]["label"],
                "n": n_samp,
                "Cal. rate": f"{m.get('cal_rate', np.nan):.3f}",
                "Median ε̂": f"{m.get('median_eps', np.nan):.4f}",
                "Feas. Gen.": f"{m.get('feas_general', np.nan):.3f}",
                "Feas. Sym.": f"{m.get('feas_symmetric', np.nan):.3f}",
                "Feas. SU": f"{m.get('feas_sym_unimodal', np.nan):.3f}",
                "Cov. Gen.": f"{m.get('coverage_general', np.nan):.3f}",
                "Cov. Sym.": f"{m.get('coverage_symmetric', np.nan):.3f}",
                "Cov. SU": f"{m.get('coverage_sym_unimodal', np.nan):.3f}",
                "WC-ES Gen.": f"{m.get('mean_wc_general', np.nan):.4f}",
                "WC-ES Sym.": f"{m.get('mean_wc_symmetric', np.nan):.4f}",
                "WC-ES SU": f"{m.get('mean_wc_sym_unimodal', np.nan):.4f}",
                "N rep.": n_outer,
            }
            table_rows.append(row)
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(RESULTS_DIR / "exp2_summary_table.csv", index=False)
    print(f"  Saved: exp2_summary_table.csv")
    print("\n--- Table 2 (n=1000) ---")
    print(table_df[table_df["n"] == 1000].to_string(index=False))

    # Figure 4 equivalent
    try:
        plot_coverage_figure(all_metrics, n_outer, RESULTS_DIR / "exp2_coverage_fig4.png")
    except Exception as exc:
        print(f"  Figure 4 skipped: {exc}")

    print("\nExp 2 complete. Results saved to results/")


if __name__ == "__main__":
    main()

"""
Experiment 1: Fixed-reference numerical comparison of worst-case ES, RVaR, and VaR.

Reproduces Figures 1, 2, 3 (and the embedded numerical tables) from paper Section 5.1.

  - ES_{0.975}   under four distributional uncertainty classes × four reference distributions.
  - RVaR_{(0.975, 0.99)} under the same four classes × four references.
  - VaR_{0.99}   under general and symmetric classes × four references.

The four uncertainty classes are:
  (a) General          — no shape restriction
  (b) Symmetric        — antisymmetric quantile (paper Section 3)
  (c) Unimodal ξ=1/2   — concave on [0,0.5], convex on [0.5,1] (Liu et al. 2025)
  (d) Symmetric unimodal — (b) + (c) (paper Theorem 4.1, Proposition 4.4)

All bounds are evaluated on a SOCP discretisation grid (paper Section 5, p. 20).

Outputs saved to results/:
  exp1_es_table.csv, exp1_rvar_table.csv, exp1_var_table.csv
  exp1_es_fig1.png, exp1_rvar_fig2.png, exp1_var_fig3.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Ensure project root is on the path when run as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ES_ALPHA, RVAR_ALPHA, RVAR_BETA, VAR_ALPHA,
    MU, SIGMA, EPS_GRID_ES, EPS_GRID_VAR, SOCP_GRID_N,
)
from src.reference_distributions import (
    REFERENCE_DISTRIBUTIONS, compute_moments,
)
from src.socp_bounds import worst_case_socp
from src.symmetric_bounds import (          # Corollaries 3.1–3.3
    worst_case_es_symmetric,
    worst_case_rvar_symmetric,
    worst_case_var_symmetric,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Classes and their display labels
CLASSES_ES_RVAR = [
    ("general",      "General"),
    ("symmetric",    "Symmetric"),
    ("unimodal",     "Unimodal (ξ=1/2)"),
    ("sym_unimodal", "Sym. unimodal"),
]
CLASSES_VAR = [
    ("general",   "General"),
    ("symmetric", "Symmetric"),
]

REF_NAMES = ["uniform", "normal", "lognormal", "pareto"]
REF_PANEL_LABELS = ["(a) Uniform", "(b) Normal", "(c) Lognormal", "(d) Pareto"]

# Colours matching the paper's style
COLOURS = {"general": "#1f77b4", "symmetric": "#ff7f0e",
           "unimodal": "#2ca02c", "sym_unimodal": "#d62728"}
LINESTYLES = {"general": "-", "symmetric": "--",
              "unimodal": "-.", "sym_unimodal": ":"}


def compute_bounds(
    risk_measure: str,
    n_socp: int,
) -> dict:
    """
    Compute worst-case bounds for all references, classes, and epsilon values.

    Returns a nested dict: results[ref_name][class_name][eps] = value or None.
    """
    eps_grid = EPS_GRID_ES if risk_measure in ("ES", "RVaR") else EPS_GRID_VAR
    classes = CLASSES_ES_RVAR if risk_measure in ("ES", "RVaR") else CLASSES_VAR

    results: dict = {ref: {cls: {} for cls, _ in classes} for ref in REF_NAMES}

    total_calls = len(REF_NAMES) * len(classes) * len(eps_grid)
    done = 0
    t0 = time.time()

    for ref_name in REF_NAMES:
        ref = REFERENCE_DISTRIBUTIONS[ref_name]
        q_fn = ref["quantile_fn"]
        mu_F, sigma_F = ref["moments_fn"]()

        for cls_name, _ in classes:
            for eps in eps_grid:
                eps_f = float(eps)
                if cls_name == "symmetric":
                    # Analytical: Corollaries 3.1–3.3 (paper p. 11–15)
                    if risk_measure == "ES":
                        val = worst_case_es_symmetric(
                            q_fn, eps_f, ES_ALPHA, MU, SIGMA, mu_F, sigma_F)
                    elif risk_measure == "RVaR":
                        val = worst_case_rvar_symmetric(
                            q_fn, eps_f, RVAR_ALPHA, RVAR_BETA,
                            MU, SIGMA, mu_F, sigma_F)
                    else:  # VaR
                        val = worst_case_var_symmetric(
                            q_fn, eps_f, VAR_ALPHA, MU, SIGMA, mu_F, sigma_F)
                else:
                    # SOCP (paper Eq. 39) for general, unimodal, sym_unimodal
                    val = worst_case_socp(
                        quantile_fn=q_fn,
                        eps=eps_f,
                        risk_measure=risk_measure,
                        kind=cls_name,
                        mu=MU,
                        sigma=SIGMA,
                        mu_F=mu_F,
                        sigma_F=sigma_F,
                        alpha=ES_ALPHA if risk_measure != "VaR" else VAR_ALPHA,
                        beta=RVAR_BETA,
                        n=n_socp,
                    )
                results[ref_name][cls_name][eps_f] = val
                done += 1
                if done % 20 == 0 or done == total_calls:
                    elapsed = time.time() - t0
                    print(
                        f"  {risk_measure}: {done}/{total_calls} done "
                        f"({elapsed:.0f}s elapsed)", flush=True
                    )

    return results


def build_table(results: dict, eps_grid: np.ndarray, classes: list) -> pd.DataFrame:
    """Convert nested results dict to a wide DataFrame."""
    rows = []
    for ref_name in REF_NAMES:
        for eps in eps_grid:
            row = {"ref": ref_name, "eps": float(eps)}
            for cls_name, _ in classes:
                row[cls_name] = results[ref_name][cls_name].get(float(eps), None)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_figure(
    results: dict,
    eps_grid: np.ndarray,
    classes: list,
    title: str,
    ylabel: str,
    save_path: Path,
) -> None:
    """Create a 2×2 subplot figure matching the paper's Figures 1/2/3 layout."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()

    for ax_idx, (ref_name, panel_label) in enumerate(zip(REF_NAMES, REF_PANEL_LABELS)):
        ax = axes[ax_idx]
        for cls_name, cls_label in classes:
            vals = [results[ref_name][cls_name].get(float(e)) for e in eps_grid]
            eps_plot, val_plot = [], []
            for e, v in zip(eps_grid, vals):
                if v is not None:
                    eps_plot.append(e)
                    val_plot.append(v)
            if eps_plot:
                ax.plot(
                    eps_plot, val_plot,
                    label=cls_label,
                    color=COLOURS.get(cls_name, "black"),
                    linestyle=LINESTYLES.get(cls_name, "-"),
                    linewidth=1.8,
                )
        ax.set_title(panel_label, fontsize=11)
        ax.set_xlabel("ε (squared Wasserstein budget)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: Fixed-reference bounds")
    parser.add_argument("--n-socp", type=int, default=SOCP_GRID_N,
                        help="SOCP grid resolution")
    args = parser.parse_args()

    n_socp = args.n_socp
    print(f"Exp 1: SOCP grid size = {n_socp}")

    # --- ES ---
    print("Computing ES bounds...")
    es_results = compute_bounds("ES", n_socp)
    es_df = build_table(es_results, EPS_GRID_ES, CLASSES_ES_RVAR)
    es_df.to_csv(RESULTS_DIR / "exp1_es_table.csv", index=False)
    plot_figure(
        es_results, EPS_GRID_ES, CLASSES_ES_RVAR,
        title=f"Worst-case ES_{{{ES_ALPHA}}} under moment and Wasserstein constraints (Fig. 1)",
        ylabel=f"WC-ES$_{{{ES_ALPHA}}}$",
        save_path=RESULTS_DIR / "exp1_es_fig1.png",
    )

    # --- RVaR ---
    print("Computing RVaR bounds...")
    rvar_results = compute_bounds("RVaR", n_socp)
    rvar_df = build_table(rvar_results, EPS_GRID_ES, CLASSES_ES_RVAR)
    rvar_df.to_csv(RESULTS_DIR / "exp1_rvar_table.csv", index=False)
    plot_figure(
        rvar_results, EPS_GRID_ES, CLASSES_ES_RVAR,
        title=(f"Worst-case RVaR$_{{({RVAR_ALPHA},{RVAR_BETA})}}$ "
               "under moment and Wasserstein constraints (Fig. 2)"),
        ylabel=f"WC-RVaR$_{{{RVAR_ALPHA},{RVAR_BETA}}}$",
        save_path=RESULTS_DIR / "exp1_rvar_fig2.png",
    )

    # --- VaR ---
    print("Computing VaR bounds...")
    var_results = compute_bounds("VaR", n_socp)
    var_df = build_table(var_results, EPS_GRID_VAR, CLASSES_VAR)
    var_df.to_csv(RESULTS_DIR / "exp1_var_table.csv", index=False)
    plot_figure(
        var_results, EPS_GRID_VAR, CLASSES_VAR,
        title=f"Worst-case VaR$_{{{VAR_ALPHA}}}$ under moment and Wasserstein constraints (Fig. 3)",
        ylabel=f"WC-VaR$_{{{VAR_ALPHA}}}$",
        save_path=RESULTS_DIR / "exp1_var_fig3.png",
    )

    # Print summary table for paper comparison (ES, uniform reference)
    print("\n--- ES_{0.975} for Uniform reference (comparison with paper Fig. 1(a)) ---")
    uni_es = es_df[es_df["ref"] == "uniform"][["eps"] + [c for c, _ in CLASSES_ES_RVAR]]
    print(uni_es.to_string(index=False, float_format="{:.2f}".format))

    print("\n--- RVaR_{(0.975, 0.99)} for Uniform reference ---")
    uni_rvar = rvar_df[rvar_df["ref"] == "uniform"][["eps"] + [c for c, _ in CLASSES_ES_RVAR]]
    print(uni_rvar.to_string(index=False, float_format="{:.2f}".format))

    print("\nExp 1 complete. Results saved to results/")


if __name__ == "__main__":
    main()

"""
Configuration for worst-case distortion riskmetrics experiments.
Paper: "Robust distortion riskmetrics under symmetry, unimodality, and Wasserstein constraints"
       Liu, Zhang, Zuo (2026).

All parameters defined at module level.  One comment per variable: meaning, unit, source.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Risk measure levels (Section 5.1)
# ---------------------------------------------------------------------------
ES_ALPHA = 0.975        # ES probability level; dimensionless; paper Section 5.1
RVAR_ALPHA = 0.975      # RVaR lower probability level; dimensionless; paper Section 5.1
RVAR_BETA = 0.99        # RVaR upper probability level; dimensionless; paper Section 5.1
VAR_ALPHA = 0.99        # VaR probability level; dimensionless; paper Section 5.1

# ---------------------------------------------------------------------------
# Target moments for candidate distributions (Section 5)
# ---------------------------------------------------------------------------
MU = 1.0                # Target mean; same units as reference distributions; paper Section 5.1
SIGMA = 1.0             # Target standard deviation; same units; paper Section 5.1

# ---------------------------------------------------------------------------
# Wasserstein budget grids (Section 5.1)
# Paper uses squared 2-Wasserstein budget ε.
# ---------------------------------------------------------------------------
# ES and RVaR: ε ∈ {0.0, 0.2, 0.4, ..., 1.6}
EPS_GRID_ES = np.arange(0.0, 1.601, 0.2)   # squared Wasserstein budgets for ES/RVaR; paper Fig. 1/2
# VaR: ε ∈ {0.0, 0.1, 0.2, ..., 1.8}
EPS_GRID_VAR = np.arange(0.0, 1.801, 0.1)  # squared Wasserstein budgets for VaR; paper Fig. 3

# ---------------------------------------------------------------------------
# SOCP grid for exp_1 (quantile discretisation, paper Section 5)
# ---------------------------------------------------------------------------
SOCP_GRID_N = 2000      # number of quantile grid points in [0,1] for exp_1 SOCP; not specified in paper

# ---------------------------------------------------------------------------
# Symmetric unimodal grid for exp_2 (paper Section 5.2, p. 24-25)
# Grid on [0.5, 1] has 131 nodes:
#   40 equal-spaced intervals on [0.5, 0.9]   → 41 nodes
#   40 equal-spaced intervals on [0.9, 0.975]  → 40 new nodes
#   48 geometrically-spaced intervals [0.975, 1-1e-5] → 48 new nodes
#   node at 1                                  →  1 new node
#   node at 0.9625 (hinge of sym-uni ES optimizer at α=0.975 under moments alone) → 1 new node
# Total: 131 nodes
# ---------------------------------------------------------------------------
def _build_exp2_grid() -> np.ndarray:
    """Build the 131-node upper-half grid for exp_2 symmetric unimodal SOCP (paper p. 24-25)."""
    seg1 = np.linspace(0.5, 0.9, 41)                            # 41 nodes
    seg2 = np.linspace(0.9, 0.975, 41)[1:]                      # 40 new nodes (drops 0.9)
    # 48 geometrically spaced intervals from 0.975 to 1-1e-5 → 49 nodes total,
    # but 0.975 is already in seg2, so we keep only the 48 new nodes above 0.975.
    # Build in complement space [0.025, 1e-5] (geomspace), then convert back.
    seg3_comp = np.geomspace(0.025, 1e-5, 49)   # 49 values decreasing: 0.025 → 1e-5
    seg3 = 1.0 - seg3_comp                       # 49 values increasing: 0.975 → 0.99999
    seg3 = seg3[1:]                               # 48 new nodes (drops 0.975, keeps up to 0.99999)
    extra = np.array([1.0, 0.9625])               # 2 new nodes; 0.9625 is hinge of sym-uni optimizer
    grid = np.unique(np.concatenate([seg1, seg2, seg3, extra]))
    assert len(grid) == 131, f"Expected 131 nodes, got {len(grid)}"
    return grid


EXP2_US_GRID = _build_exp2_grid()   # 131-node upper-half grid for symmetric unimodal; paper p. 24-25

# ---------------------------------------------------------------------------
# Simulation parameters (Section 5.2)
# ---------------------------------------------------------------------------
N_OUTER_PAPER = 1000        # outer simulation repetitions in paper; paper Section 5.2
N_OUTER_RUN = 50            # outer repetitions actually executed (runtime reduction); estimated >8 h for full
N_BOOTSTRAP = 500           # bootstrap resamples per repetition; paper Section 5.2
SAMPLE_SIZES = [250, 500, 1000, 2000]   # sample sizes; paper Section 5.2
COVERAGE_QUANTILE = 0.95    # bootstrap quantile used for calibration (475th of 500); paper Section 5.2
N_CALIBRATION_ORDER = 475   # 475th order statistic of B=500 required budgets; paper Section 5.2

# Computational tolerance for SOCP solver
SOLVER_TOL = 1e-8           # solver absolute tolerance; not specified in paper

# True population ES for each sampling distribution (α=0.975), pre-computed
# For exp_2 coverage computation
# Normal N(0,1): ES = phi(qnorm(0.975))/0.025 = phi(1.96)/0.025
# sqrt(3/5)*T5: ES = sqrt(3/5) * ES_{T5}(0.975)
# Laplace(0, 1/sqrt(2)): computed analytically
# Bimodal and Lognormal: computed numerically
POPULATION_ES_ALPHA = 0.975  # ES level for exp_2; paper Section 5.2

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42    # not specified in paper; chosen for reproducibility

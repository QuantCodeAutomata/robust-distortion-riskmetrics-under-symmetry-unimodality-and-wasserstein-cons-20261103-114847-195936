"""
Bootstrap calibration of squared Wasserstein budget and worst-case ES evaluation.

Implements the simulation study of paper Section 5.2:
  - Fast O(n) analytical formula for the general-class ES bound at each Wasserstein budget.
  - Required-budget computation for each bootstrap resample.
  - Calibrated budget as the 475th order statistic of 500 required budgets.
  - Evaluation of general, symmetric, and symmetric-unimodal worst-case ES at calibrated budget.

The fast analytical formula is derived from the closed-form SOCP solution for the general class
when the reference distribution is empirical and the score is monotone for lambda >= 0.
See src/socp_bounds.py for the SOCP formulation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import Callable

from src.config import (
    MU, SIGMA, N_CALIBRATION_ORDER, POPULATION_ES_ALPHA,
    EXP2_US_GRID, SOLVER_TOL,
)
from src.socp_bounds import worst_case_sym_unimodal_empirical
from src.symmetric_bounds import (
    worst_case_es_symmetric,
    feasibility_threshold,
)


# ---------------------------------------------------------------------------
# Empirical ES
# ---------------------------------------------------------------------------

def empirical_es(sample: np.ndarray, alpha: float) -> float:
    """
    Empirical ES at level alpha from a sample.

    Uses exact partial weighting for the tail boundary (paper Section 5.2).
    ES_alpha = (1/(1-alpha)) * integral_alpha^1 F_n^{-1}(u) du.
    """
    xs = np.sort(sample)
    n = len(xs)
    # Empirical quantile at u: F_n^{-1}(u) = x_{(ceil(n*u))}
    # Integral_alpha^1 F_n^{-1}(u) du using exact piecewise-constant representation:
    # F_n^{-1}(u) = x_{(i)} for u in ((i-1)/n, i/n]
    # Integral = sum_{i=1}^n x_{(i)} * max(0, min(i/n, 1) - max((i-1)/n, alpha))
    tail_mass = 0.0
    tail_sum = 0.0
    for i in range(n):
        lo = i / n
        hi = (i + 1) / n
        overlap = max(0.0, min(hi, 1.0) - max(lo, alpha))
        if overlap > 0:
            tail_mass += overlap
            tail_sum += xs[i] * overlap
    if tail_mass < 1e-12:
        return xs[-1]
    return tail_sum / tail_mass


# ---------------------------------------------------------------------------
# Fast analytical general-class ES bound (O(1) per lambda, O(n) setup)
# ---------------------------------------------------------------------------

class _GeneralESState:
    """
    Pre-computed statistics for the fast analytical general-class ES formula.

    For empirical reference F_n with sorted order statistics x_{(1)} <= ... <= x_{(n)},
    target moments (mu, sigma), and ES level alpha:

      WC_ES(lambda) = mu + sigma * (C + lambda*A) / sqrt(C + lambda^2*B + 2*lambda*A)

    where:
      A = ES_alpha(F_n) - mu_n  (excess tail mean over sample mean)
      B = Var(F_n)               (empirical variance)
      C = alpha / (1 - alpha)    (ES weight ratio)
      A_F = (mu - mu_n)^2 + sigma^2 + B  (Wasserstein diameter)

    The formula assumes the score is already nondecreasing, which holds for lambda >= 0.

    The required budget corresponding to target WC_ES = v is:
      epsilon_b^req = A_F - 2 * sigma * (A + lambda_v * B) / sqrt(C + lambda_v^2*B + 2*lambda_v*A)

    where lambda_v satisfies (C + lambda_v*A)/sqrt(C + lambda_v^2*B + 2*lambda_v*A) = (v-mu)/sigma.
    """

    def __init__(
        self, sorted_sample: np.ndarray, alpha: float, mu: float, sigma: float
    ) -> None:
        n = len(sorted_sample)
        mu_n = float(np.mean(sorted_sample))
        var_n = float(np.var(sorted_sample))   # ddof=0

        # Index of the alpha quantile: empirical quantile F_n^{-1}(alpha)
        # Using exact integral for ES
        es_n = empirical_es(sorted_sample, alpha)

        self.mu_n = mu_n
        self.var_n = var_n
        self.es_n = es_n
        self.A = es_n - mu_n           # excess tail mean
        self.B = var_n                  # empirical variance
        self.C = alpha / (1.0 - alpha)  # ES weight ratio
        self.A_F = (mu - mu_n) ** 2 + sigma ** 2 + var_n
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha

    def wc_es_at_lambda(self, lam: float) -> float:
        """WC_ES_general(lambda): WC ES as function of Lagrange multiplier lambda."""
        denom_sq = self.C + lam ** 2 * self.B + 2.0 * lam * self.A
        if denom_sq <= 0:
            return self.mu + self.sigma * np.sqrt(self.C)
        return self.mu + self.sigma * (self.C + lam * self.A) / np.sqrt(denom_sq)

    def eps_at_lambda(self, lam: float) -> float:
        """Squared Wasserstein distance d^2_W(F_n, H_lambda) as function of lambda."""
        denom_sq = self.C + lam ** 2 * self.B + 2.0 * lam * self.A
        if denom_sq <= 0:
            return self.A_F
        return self.A_F - 2.0 * self.sigma * (self.A + lam * self.B) / np.sqrt(denom_sq)

    def moment_only_bound(self) -> float:
        """Worst-case ES under moment constraints only (lambda = 0)."""
        return self.mu + self.sigma * np.sqrt(self.C)

    def es_reference(self) -> float:
        """ES_alpha(F_n) = WC_ES as lambda -> infinity."""
        if self.B < 1e-12:
            return self.mu + self.A
        return self.mu + self.sigma * self.A / np.sqrt(self.B)

    def required_budget(self, target_v: float) -> float:
        """
        Compute epsilon^req_b = inf{epsilon >= 0 : WC_ES_gen(epsilon) >= target_v}.

        Returns inf (np.inf) if target_v exceeds the moment-only bound,
        and 0.0 if target_v <= ES_alpha(F_n).
        """
        v_max = self.moment_only_bound()
        v_min = self.es_reference()

        if target_v >= v_max - 1e-10:
            return np.inf  # beyond moment-only bound → calibration failure

        if target_v <= v_min + 1e-10:
            return max(0.0, self.eps_at_lambda(1e8))  # essentially eps = 0

        # WC_ES is decreasing in lambda, so target c* = (target_v - mu)/sigma
        c_star = (target_v - self.mu) / self.sigma

        def obj(lam: float) -> float:
            denom_sq = self.C + lam ** 2 * self.B + 2.0 * lam * self.A
            if denom_sq <= 0:
                return np.sqrt(self.C) - c_star
            return (self.C + lam * self.A) / np.sqrt(denom_sq) - c_star

        # At lam=0: obj = sqrt(C) - c* > 0 (since c* < sqrt(C))
        # As lam→∞: obj → A/sqrt(B) - c* < 0 (since c* > A/sqrt(B))
        lam_hi = 1.0
        for _ in range(50):
            if obj(lam_hi) < 0:
                break
            lam_hi *= 2.0

        try:
            lam_star = brentq(obj, 0.0, lam_hi, xtol=1e-10, maxiter=200)
        except ValueError:
            return np.inf

        return float(self.eps_at_lambda(lam_star))


# ---------------------------------------------------------------------------
# Bootstrap calibration for one outer repetition
# ---------------------------------------------------------------------------

def bootstrap_calibrate(
    sample: np.ndarray,
    alpha: float,
    mu: float,
    sigma: float,
    n_bootstrap: int,
    calibration_order: int,
    rng: np.random.Generator,
) -> tuple[float, bool]:
    """
    Bootstrap calibration of the squared Wasserstein budget (paper Section 5.2).

    For each of B=n_bootstrap resamples:
      1. Draw training resample F_n^{*(b)} with replacement.
      2. Compute out-of-bag validation ES: V_b.
      3. Compute required budget epsilon_b^req using the fast general-class formula.

    The calibrated budget is the calibration_order-th order statistic of the
    required budgets (paper: 475th of 500).

    Parameters
    ----------
    sample : 1-D array
        Observed sample X_1, ..., X_n.
    alpha : float
        ES probability level (0.975 in exp_2).
    mu, sigma : float
        Known population moments (treated as fixed, not re-estimated; paper p. 24).
    n_bootstrap : int
        Number of bootstrap resamples B.
    calibration_order : int
        Order statistic index (0-based) for the calibrated budget (e.g., 474 for 475th).
    rng : numpy Generator
        Random number generator.

    Returns
    -------
    (eps_hat, success): calibrated budget and whether calibration succeeded.
    """
    n = len(sample)
    required_budgets: list[float] = []

    for _ in range(n_bootstrap):
        # Draw bootstrap resample (with replacement)
        boot_idx = rng.integers(0, n, size=n)
        boot_sample = sample[boot_idx]
        sorted_boot = np.sort(boot_sample)

        # Out-of-bag indices
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False
        oob_sample = sample[oob_mask]

        if len(oob_sample) < 5:
            # Too few OOB samples — skip (treat as requiring infinite budget)
            required_budgets.append(np.inf)
            continue

        # Validation ES from OOB observations
        v_b = empirical_es(oob_sample, alpha)

        # Fast general-class required budget
        state = _GeneralESState(sorted_boot, alpha, mu, sigma)
        req = state.required_budget(v_b)
        required_budgets.append(req)

    # Calibrated budget: calibration_order-th order statistic (0-indexed)
    sorted_reqs = np.sort(required_budgets)
    eps_hat = sorted_reqs[calibration_order]

    success = np.isfinite(eps_hat)
    return float(eps_hat) if success else np.inf, success


# ---------------------------------------------------------------------------
# Post-calibration evaluation
# ---------------------------------------------------------------------------

def evaluate_all_classes(
    sample: np.ndarray,
    eps_hat: float,
    alpha: float,
    mu: float,
    sigma: float,
    upper_half_grid: np.ndarray,
    mu_F: float | None = None,
    sigma_F: float | None = None,
    solver_tol: float = SOLVER_TOL,
) -> dict[str, float | None]:
    """
    Evaluate worst-case ES for all three uncertainty classes at the calibrated budget.

    Parameters
    ----------
    sample : 1-D array
        Full observed sample.
    eps_hat : float
        Calibrated squared Wasserstein budget.
    alpha : float
        ES probability level.
    mu, sigma : float
        Known population moments.
    upper_half_grid : 1-D array
        131-node grid for symmetric unimodal SOCP.
    mu_F, sigma_F : float or None
        If None, use sample mean and std.
    solver_tol : float
        SOCP solver tolerance.

    Returns
    -------
    dict with keys 'general', 'symmetric', 'sym_unimodal', values float or None.
    """
    xs = np.sort(sample)
    mu_n = float(np.mean(xs)) if mu_F is None else mu_F
    sigma_n = float(np.std(xs, ddof=0)) if sigma_F is None else sigma_F

    results: dict[str, float | None] = {}

    # General class: fast analytical formula
    state = _GeneralESState(xs, alpha, mu, sigma)
    # Find lambda for eps_hat
    A_F = state.A_F
    if eps_hat >= A_F:
        results['general'] = state.moment_only_bound()
    else:
        # Find lambda s.t. eps_at_lambda(lam) = eps_hat
        def obj_lam(lam: float) -> float:
            return state.eps_at_lambda(lam) - eps_hat

        try:
            lam_hi = 1.0
            for _ in range(50):
                if obj_lam(lam_hi) < 0:
                    break
                lam_hi *= 2.0
            lam_star = brentq(obj_lam, 0.0, lam_hi, xtol=1e-10, maxiter=200)
            results['general'] = state.wc_es_at_lambda(lam_star)
        except ValueError:
            results['general'] = state.moment_only_bound()

    # Symmetric class: closed-form from Corollary 3.1
    def quantile_fn_empirical(u: np.ndarray) -> np.ndarray:
        ecdf_u = (np.arange(1, len(xs) + 1) - 0.5) / len(xs)
        return np.interp(np.clip(u, 1e-10, 1.0 - 1e-10), ecdf_u, xs)

    try:
        sym_val = worst_case_es_symmetric(
            quantile_fn_empirical, eps_hat, alpha, mu, sigma, mu_n, sigma_n
        )
        results['symmetric'] = sym_val
    except Exception:
        results['symmetric'] = None

    # Symmetric unimodal: SOCP on 131-node grid
    try:
        sym_uni_val = worst_case_sym_unimodal_empirical(
            sample, eps_hat, alpha, mu, sigma, upper_half_grid, solver_tol=solver_tol
        )
        results['sym_unimodal'] = sym_uni_val
    except Exception:
        results['sym_unimodal'] = None

    return results

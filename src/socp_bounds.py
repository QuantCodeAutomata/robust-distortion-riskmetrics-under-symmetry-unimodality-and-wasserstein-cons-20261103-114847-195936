"""
SOCP-based worst-case distortion riskmetric bounds.

Solves the second-order cone program (paper Eq. 39) for four uncertainty classes:
  - General (no shape restriction)
  - Unimodal with quantile concavity-convexity transition at xi=1/2 (Liu et al. 2025)
  - Symmetric unimodal (paper Theorem 4.1, Proposition 4.4)

Also provides the general-class bound for an empirical (discrete) reference,
used in the bootstrap calibration of Section 5.2.

All SOCPs are solved with CVXPY + Clarabel (paper Section 5, p. 20).
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Callable

try:
    import clarabel  # noqa: F401  # ensure Clarabel is available
    _SOLVER = cp.CLARABEL
except ImportError:
    _SOLVER = cp.SCS  # fallback


# ---------------------------------------------------------------------------
# Quadrature helpers
# ---------------------------------------------------------------------------

def _uniform_grid(n: int) -> np.ndarray:
    """Midpoint rule grid on (0, 1) with n points."""
    return (np.arange(n) + 0.5) / n


def _trapezoid_weights(u: np.ndarray) -> np.ndarray:
    """Integration weights for trapezoidal rule on sorted u in (0,1)."""
    n = len(u)
    if n == 1:
        return np.array([1.0])
    w = np.zeros(n)
    w[0] = (u[1] - 0.0) / 2.0
    w[-1] = (1.0 - u[-2]) / 2.0
    w[1:-1] = (u[2:] - u[:-2]) / 2.0
    return w


def _d_epsilon(
    mu: float, sigma: float, mu_F: float, sigma_F: float, eps: float
) -> float:
    """
    D_epsilon = ((mu - mu_F)^2 + sigma^2 + sigma_F^2 - epsilon) / 2
    Paper Eq. (38), p. 19.
    """
    return 0.5 * ((mu - mu_F) ** 2 + sigma ** 2 + sigma_F ** 2 - eps)


# ---------------------------------------------------------------------------
# Weight vector for the SOCP objective
# ---------------------------------------------------------------------------

def _es_weights(u: np.ndarray, alpha: float, du: float) -> np.ndarray:
    """
    Objective weights for ES_alpha: w(u) = g'(1-u) = 1/(1-alpha) for u > alpha else 0.
    Paper: distortion g(t) = min(t/(1-alpha), 1), g'(t) = 1/(1-alpha)*1_{t<1-alpha}.
    Then w(u) = g'(1-u) = 1/(1-alpha)*1_{u>alpha}.
    """
    w = np.where(u > alpha, 1.0 / (1.0 - alpha), 0.0)
    return w * du


def _rvar_weights(u: np.ndarray, alpha: float, beta: float, du: float) -> np.ndarray:
    """
    Objective weights for RVaR_{alpha,beta}: w(u) = g'(1-u) = 1/(beta-alpha) for u in (alpha, beta).
    For RVaR distortion g'(t)=1/(beta-alpha) for t in (1-beta,1-alpha), so g'(1-u) is nonzero
    for 1-u in (1-beta,1-alpha), i.e. u in (alpha,beta).  # paper Eq. 39 objective
    """
    w = np.where((u > alpha) & (u < beta), 1.0 / (beta - alpha), 0.0)
    return w * du


def _var_idx(u: np.ndarray, alpha: float) -> int:
    """Index of the grid point closest to alpha for VaR."""
    return int(np.argmin(np.abs(u - alpha)))


# ---------------------------------------------------------------------------
# Shared SOCP solver
# ---------------------------------------------------------------------------

def _solve_socp(
    objective_w: np.ndarray,
    u: np.ndarray,
    du: float,
    F_inv: np.ndarray,
    D_eps_sigma: float,
    kind: str,
    var_idx: int | None = None,
    solver_tol: float = 1e-7,
) -> tuple[float | None, bool]:
    """
    Solve the SOCP (paper Eq. 39) for a given uncertainty class.

    Parameters
    ----------
    objective_w : 1-D array
        Weight vector for the objective: max sum(objective_w * y).
        For ES/RVaR: objective_w = w(u)*du; for VaR: unused (var_idx used instead).
    u : 1-D array
        Quantile grid in (0,1).
    du : float
        Uniform quadrature weight (= 1/N).
    F_inv : 1-D array
        Reference quantile values at grid points.
    D_eps_sigma : float
        D_epsilon / sigma (Wasserstein constraint RHS).
    kind : str
        One of 'general', 'unimodal', 'sym_unimodal'.
    var_idx : int or None
        Grid index for VaR objective (used when objective_w is None).
    solver_tol : float
        Absolute tolerance for Clarabel.

    Returns
    -------
    (value, feasible): worst-case risk value (or None), feasibility flag.
    """
    N = len(u)
    half = N // 2

    if kind in ('symmetric', 'sym_unimodal'):
        # Parameterize upper half [0.5, 1] only; antisymmetric extension.
        # y_up: N - half values (upper half); y_lo = -y_up[::-1]
        y_up = cp.Variable(N - half, nonneg=False)
        # Full vector: y = [-y_up[::-1], y_up]
        constraints = [
            cp.diff(y_up) >= 0,          # nondecreasing on [0.5, 1]
        ]
        if kind == 'sym_unimodal':
            constraints.append(cp.diff(y_up, 2) >= 0)   # convex on [0.5, 1]
        # Build full y
        y_full = cp.hstack([-y_up[::-1], y_up])

        # Norm constraint: ||y||^2 = 2 * ||y_up||^2 <= 1 (since antisymmetric)
        constraints.append(2.0 * cp.sum_squares(y_up) * du <= 1.0)

        # Centered: sum(y_full) = 0 (automatic by antisymmetry — skip for speed)
        # But add explicitly in case of numerical issues:
        constraints.append(cp.sum(y_full) * du == 0.0)

        # Wasserstein: <F_inv, y_full> >= D_eps_sigma
        wass_inner = cp.sum(cp.multiply(F_inv, y_full)) * du
        constraints.append(wass_inner >= D_eps_sigma)

        # Objective
        if var_idx is None:
            obj = cp.Maximize(cp.sum(cp.multiply(objective_w, y_full)))
        else:
            # VaR: objective is y at index var_idx
            obj = cp.Maximize(y_full[var_idx])

        prob = cp.Problem(obj, constraints)

    else:
        # General and Unimodal: full-length y
        y = cp.Variable(N)
        constraints = [
            cp.diff(y) >= 0,                         # nondecreasing
            cp.sum(y) * du == 0.0,                   # centered (mean = 0)
            cp.sum_squares(y) * du <= 1.0,           # unit L^2 norm
            cp.sum(cp.multiply(F_inv, y)) * du >= D_eps_sigma,  # Wasserstein
        ]
        if kind == 'unimodal':
            # Concave on [0, 0.5] and convex on [0.5, 1]
            constraints.append(cp.diff(y[:half + 1], 2) <= 0)  # concave lower
            constraints.append(cp.diff(y[half:], 2) >= 0)       # convex upper

        # Objective
        if var_idx is None:
            obj = cp.Maximize(cp.sum(cp.multiply(objective_w, y)))
        else:
            obj = cp.Maximize(y[var_idx])

        prob = cp.Problem(obj, constraints)
        y_full = y  # alias

    # Solve
    solver_opts = {"eps_abs": solver_tol, "eps_rel": solver_tol}
    try:
        prob.solve(solver=_SOLVER, solver_opts=solver_opts, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            return None, False

    if prob.status in ("infeasible", "infeasible_inaccurate", "unbounded"):
        return None, False

    if prob.status in ("optimal", "optimal_inaccurate"):
        if var_idx is None:
            val = float(prob.value) if prob.value is not None else None
        else:
            val = float(y_full.value[var_idx]) if y_full.value is not None else None
        return val, True

    return None, False


# ---------------------------------------------------------------------------
# Public interface for exp_1
# ---------------------------------------------------------------------------

def worst_case_socp(
    quantile_fn: Callable,
    eps: float,
    risk_measure: str,
    kind: str,
    mu: float,
    sigma: float,
    mu_F: float,
    sigma_F: float,
    alpha: float = 0.975,
    beta: float = 0.99,
    n: int = 1500,
    solver_tol: float = 1e-7,
) -> float | None:
    """
    Worst-case distortion riskmetric via the SOCP (paper Eq. 39, p. 20).

    Parameters
    ----------
    quantile_fn : callable
        Reference distribution quantile function F^{-1}.
    eps : float
        Squared 2-Wasserstein budget epsilon.
    risk_measure : str
        One of 'ES', 'RVaR', 'VaR'.
    kind : str
        One of 'general', 'unimodal', 'sym_unimodal'.
    mu, sigma : float
        Target mean and standard deviation.
    mu_F, sigma_F : float
        Reference distribution mean and standard deviation.
    alpha : float
        Lower probability level.
    beta : float
        Upper probability level (for RVaR).
    n : int
        Quantile grid resolution.
    solver_tol : float
        SOCP solver tolerance.

    Returns
    -------
    float or None
        Worst-case value, or None if infeasible.
    """
    # At eps=0 the Wasserstein constraint forces H = F; return riskmetric of F directly.
    # The SOCP is numerically unstable when the constraint is exactly tight at eps=0.
    if eps < 1e-10:
        u_ref = _uniform_grid(10000)
        F_ref = quantile_fn(np.clip(u_ref, 1e-10, 1.0 - 1e-10))
        if kind in ("symmetric", "sym_unimodal"):
            # Feasible only if reference is symmetric (sigma_0 ≈ sigma_F; paper Lemma 3.1)
            F_flip = quantile_fn(np.clip(1.0 - u_ref, 1e-10, 1.0 - 1e-10))
            sigma0_sq = float(np.mean(((F_ref - F_flip) / 2.0) ** 2))
            if sigma0_sq < sigma ** 2 - 1e-3:
                return None
        if risk_measure == "ES":
            return float(np.mean(F_ref[u_ref > alpha]))
        elif risk_measure == "RVaR":
            mask = (u_ref > alpha) & (u_ref < beta)
            return float(np.mean(F_ref[mask])) if np.any(mask) else None
        else:  # VaR
            return float(quantile_fn(alpha))

    u = _uniform_grid(n)
    du = 1.0 / n
    F_inv = quantile_fn(np.clip(u, 1e-10, 1.0 - 1e-10))

    # Center and normalize F_inv to (F^{-1} - mu_F) / sigma_F? No — paper uses raw F^{-1}.
    # The constraint is <F^{-1}, y> >= D_eps / sigma (paper Eq. 39).
    D_eps = _d_epsilon(mu, sigma, mu_F, sigma_F, eps)
    D_eps_sigma = D_eps / sigma

    # Build objective weights
    var_idx = None
    if risk_measure == 'ES':
        objective_w = _es_weights(u, alpha, du)
    elif risk_measure == 'RVaR':
        objective_w = _rvar_weights(u, alpha, beta, du)
    elif risk_measure == 'VaR':
        objective_w = None
        var_idx = _var_idx(u, alpha)
    else:
        raise ValueError(f"Unknown risk_measure: {risk_measure}")

    val_centered, feasible = _solve_socp(
        objective_w, u, du, F_inv, D_eps_sigma, kind,
        var_idx=var_idx, solver_tol=solver_tol,
    )
    if not feasible or val_centered is None:
        return None

    # Convert from centered y to actual risk: risk = mu*g(1) + sigma * val_centered
    # For ES, RVaR, VaR: g(1) = 1
    return float(mu + sigma * val_centered)


# ---------------------------------------------------------------------------
# Symmetric unimodal SOCP on the 131-node exp_2 grid
# ---------------------------------------------------------------------------

def worst_case_sym_unimodal_empirical(
    sample: np.ndarray,
    eps: float,
    alpha: float,
    mu: float,
    sigma: float,
    upper_half_grid: np.ndarray,
    solver_tol: float = 1e-7,
) -> float | None:
    """
    Worst-case ES_alpha for the symmetric unimodal class with an empirical reference.

    Uses the finite-dimensional cone on the paper's 131-node grid (paper p. 24-25).

    Parameters
    ----------
    sample : 1-D array
        Observed sample (not necessarily sorted).
    eps : float
        Squared Wasserstein budget.
    alpha : float
        ES probability level (> 1/2).
    mu, sigma : float
        Known population mean and standard deviation.
    upper_half_grid : 1-D array
        The 131-node upper-half grid from config.EXP2_US_GRID.
    solver_tol : float
        Solver tolerance.

    Returns
    -------
    float or None
        Worst-case ES value, or None if infeasible.
    """
    xs = np.sort(sample)
    n_samp = len(xs)
    mu_F = float(np.mean(xs))
    sigma_F = float(np.std(xs, ddof=0))

    # Upper-half grid (values in [0.5, 1])
    u_up = upper_half_grid
    n_up = len(u_up)

    # Build the antisymmetric extension for the full grid
    u_lo = 1.0 - u_up[::-1]   # mirror to [0, 0.5]; reversed
    # Note: u_lo ends at 1 - u_up[0] = 1 - 0.5 = 0.5 (excluded since u_up[0] = 0.5)
    u_lo = u_lo[:-1]            # drop the 0.5 (midpoint) to avoid duplication
    u_full = np.concatenate([u_lo, u_up])  # full grid
    n_full = len(u_full)

    # Empirical quantile at grid points (linearly interpolated)
    ecdf_u = (np.arange(1, n_samp + 1) - 0.5) / n_samp  # empirical quantile levels
    F_inv_full = np.interp(u_full, ecdf_u, xs)

    # D_epsilon
    D_eps = _d_epsilon(mu, sigma, mu_F, sigma_F, eps)
    D_eps_sigma = D_eps / sigma

    # Trapezoidal weights on full grid
    w_full = _trapezoid_weights(u_full)

    # Variable: y_up (n_up values on [0.5, 1])
    # y_up[0] = 0 is the antisymmetric condition at u=0.5 (first grid point)
    # By construction u_up[0] = 0.5, so y_up[0] corresponds to u=0.5.
    # Antisymmetry forces y_up[0] = 0.
    y_up = cp.Variable(n_up, nonneg=False)
    # Full y: antisymmetric extension
    # y_lo[i] = -y_up[n_up-1 - i] for i=0,...,n_up-2  (lower half, reversed)
    y_lo_expr = -y_up[1:][::-1]   # = -y_up[n_up-1], ..., -y_up[1]
    y_full_expr = cp.hstack([y_lo_expr, y_up])

    # Weights for objective: ES_alpha
    w_obj_full = np.where(u_full > alpha, 1.0 / (1.0 - alpha), 0.0) * w_full

    constraints = [
        y_up[0] == 0.0,              # antisymmetry at u=0.5
        cp.diff(y_up) >= 0,          # nondecreasing on upper half
        cp.diff(y_up, 2) >= 0,        # convex on upper half
        # Norm: ||y_full||^2 = 2*||y_up[1:]||^2 ≈ sum(w_full * y_full^2)
        cp.sum(cp.multiply(w_full, cp.square(y_full_expr))) <= 1.0,
        # Centered (automatic by antisymmetry but add for numerical stability)
        cp.sum(cp.multiply(w_full, y_full_expr)) == 0.0,
        # Wasserstein constraint
        cp.sum(cp.multiply(F_inv_full * w_full, y_full_expr)) >= D_eps_sigma,
    ]

    obj = cp.Maximize(cp.sum(cp.multiply(w_obj_full, y_full_expr)))
    prob = cp.Problem(obj, constraints)

    solver_opts = {"eps_abs": solver_tol, "eps_rel": solver_tol}
    try:
        prob.solve(solver=_SOLVER, solver_opts=solver_opts, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            return None

    if prob.status in ("infeasible", "infeasible_inaccurate", "unbounded"):
        return None

    if prob.value is None:
        return None

    return float(mu + sigma * prob.value)


# ---------------------------------------------------------------------------
# General-class ES bound for an empirical reference (fast, for bootstrap)
# ---------------------------------------------------------------------------

def worst_case_es_general_empirical(
    sample: np.ndarray,
    eps: float,
    alpha: float,
    mu: float,
    sigma: float,
    n_socp: int = 500,
    solver_tol: float = 1e-7,
) -> float | None:
    """
    Worst-case ES_alpha for the general uncertainty class with empirical reference F_n.

    Uses a small SOCP on the empirical quantile grid (paper Section 5.2, p. 24).

    Parameters
    ----------
    sample : 1-D array
        Observed sample.
    eps : float
        Squared Wasserstein budget.
    alpha : float
        ES probability level (> 1/2).
    mu, sigma : float
        Known population mean and standard deviation.
    n_socp : int
        Grid resolution for the SOCP (should be O(n) at minimum).

    Returns
    -------
    float or None
    """
    xs = np.sort(sample)
    n_samp = len(xs)
    mu_F = float(np.mean(xs))
    sigma_F = float(np.std(xs, ddof=0))

    # Use the sorted sample points as the quantile grid, with midpoint empirical CDF
    ecdf_u = (np.arange(1, n_samp + 1) - 0.5) / n_samp
    F_inv = xs  # empirical quantile at ecdf_u levels

    # May also want a finer grid (interpolating to n_socp points)
    if n_socp > n_samp:
        u = _uniform_grid(n_socp)
        F_inv = np.interp(u, ecdf_u, xs)
        du = 1.0 / n_socp
    else:
        u = ecdf_u
        du = 1.0 / n_samp

    D_eps = _d_epsilon(mu, sigma, mu_F, sigma_F, eps)
    D_eps_sigma = D_eps / sigma
    objective_w = _es_weights(u, alpha, du)

    val, feasible = _solve_socp(
        objective_w, u, du, F_inv, D_eps_sigma, 'general',
        solver_tol=solver_tol,
    )
    if not feasible or val is None:
        return None
    return float(mu + sigma * val)

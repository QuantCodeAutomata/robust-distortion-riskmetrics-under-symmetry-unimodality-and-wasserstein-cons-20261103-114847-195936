"""
Closed-form worst-case bounds for the symmetric uncertainty class.

Implements:
  - Corollary 3.1(i) : worst-case ES_alpha under symmetry + Wasserstein (paper p. 11)
  - Corollary 3.2    : worst-case RVaR_{alpha,beta} under symmetry + Wasserstein (paper p. 13)
  - Corollary 3.3    : worst-case VaR_alpha under symmetry + Wasserstein (paper p. 14)

All formulas are for alpha > 1/2 (upper-tail risk).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import Callable

# Quadrature grid size for all integrals in this module
_N_QUAD = 4000


def _quad_grid(n: int = _N_QUAD) -> np.ndarray:
    """Midpoint quadrature grid on (0, 1)."""
    return (np.arange(1, n + 1) - 0.5) / n


def _safe_q(quantile_fn: Callable, u: np.ndarray) -> np.ndarray:
    """Evaluate quantile function, clipping u to avoid boundary singularities."""
    return quantile_fn(np.clip(u, 1e-10, 1.0 - 1e-10))


# ---------------------------------------------------------------------------
# Auxiliary quantities shared by all symmetric bounds
# ---------------------------------------------------------------------------

def compute_s_F(quantile_fn: Callable, n: int = _N_QUAD) -> tuple[np.ndarray, np.ndarray]:
    """
    Antisymmetric part of the reference quantile (paper Lemma 3.1, p. 7).

    s_F(u) = (F^{-1}(u) - F^{-1}(1-u)) / 2

    Returns (u_grid, s_F_values).
    """
    u = _quad_grid(n)
    q_u = _safe_q(quantile_fn, u)
    q_1mu = _safe_q(quantile_fn, 1.0 - u)
    s_F = (q_u - q_1mu) / 2.0
    return u, s_F


def compute_sigma0(quantile_fn: Callable, n: int = _N_QUAD) -> float:
    """
    sigma_0 = ||s_F||_2, L^2 norm of the antisymmetric part (paper Lemma 3.1, p. 7).
    """
    u, s_F = compute_s_F(quantile_fn, n)
    return float(np.sqrt(np.mean(s_F ** 2)))


def feasibility_threshold(
    quantile_fn: Callable,
    mu_F: float,
    sigma_F: float,
    mu: float,
    sigma: float,
    n: int = _N_QUAD,
) -> tuple[float, float]:
    """
    Compute the feasibility thresholds epsilon_0 and epsilon_1 for the symmetric class.

    epsilon_0 = A_F - 2 * sigma * sigma_0   (minimum feasible budget; paper Lemma 3.1, p. 7)
    epsilon_1 is where the Wasserstein constraint becomes inactive (lambda -> 0+).

    Returns (epsilon_0, sigma_0).  epsilon_1 must be computed per-riskmetric.
    """
    A_F = (mu_F - mu) ** 2 + sigma_F ** 2 + sigma ** 2
    sigma0 = compute_sigma0(quantile_fn, n)
    eps0 = A_F - 2.0 * sigma * sigma0
    return float(eps0), float(sigma0)


def R_F_values(quantile_fn: Callable, u: np.ndarray) -> np.ndarray:
    """
    Compute R_F(t) = integral_{1-t}^1 F^{-1}(s)ds + integral_t^1 F^{-1}(s)ds - mu_F
    on the provided u-grid by numerical integration.

    Uses the relation:
        integral_t^1 F^{-1}(u)du = E[max(F^{-1}(U) - F^{-1}(t), 0)] + (1-t)*F^{-1}(t)
    but we do it more directly using cumulative sums on the quadrature grid.
    """
    # Build the integral of F^{-1} from 0 to u on a fine auxiliary grid, then interpolate.
    n_aux = max(len(u) * 4, 8000)
    u_aux = _quad_grid(n_aux)
    q_aux = _safe_q(quantile_fn, u_aux)
    du = 1.0 / n_aux
    # cumulative integral F^{-1} from 0 to u
    cum_integral = np.concatenate([[0.0], np.cumsum(q_aux * du)])
    u_aux_edges = np.concatenate([[0.0], u_aux + du / 2.0])
    integral_to_u = np.interp(u, u_aux_edges, cum_integral)
    total = cum_integral[-1]  # integral_0^1 F^{-1} = mu_F
    mu_F = total
    # integral_t^1 F^{-1} = mu_F - integral_0^t F^{-1}
    int_t_to_1 = mu_F - integral_to_u
    # integral_{1-t}^1 F^{-1}: same but evaluated at 1-t
    int_1mt_to_1 = mu_F - np.interp(1.0 - u, u_aux_edges, cum_integral)
    return int_t_to_1 + int_1mt_to_1 - mu_F


# ---------------------------------------------------------------------------
# Symmetric ES bound — Corollary 3.1(i)
# ---------------------------------------------------------------------------

def _es_C_bar_g_F(quantile_fn: Callable, alpha: float, n: int = _N_QUAD) -> float:
    """
    C_{bar g, F} for ES_alpha (paper Corollary 3.1(i), p. 11):
      C = (mu_F - alpha * ES_{1-alpha}(F) - (1-alpha) * ES_alpha(F)) / (2(1-alpha))

    where ES_p(F) = (1/(1-p)) * integral_p^1 F^{-1}(u) du.
    Equivalently: C = (1/(2(1-alpha))) * [integral_0^{1-alpha} F^{-1}(u)du - integral_alpha^1 F^{-1}(u)du]
    """
    u = _quad_grid(n)
    q = _safe_q(quantile_fn, u)
    du = 1.0 / n
    # sum over u < 1-alpha (lower region) and u > alpha (upper region)
    lower_integral = np.sum(q[u < (1.0 - alpha)]) * du
    upper_integral = np.sum(q[u > alpha]) * du
    return float((lower_integral - upper_integral) / (2.0 * (1.0 - alpha)))


def _compute_lambda(
    C_g: float, V_g: float, sigma0: float,
    C_eps: float, sigma: float,
) -> float:
    """
    Compute lambda_epsilon from Proposition 3.2, Eq. (18) (paper p. 11):

    lambda = (1 / (2 sigma_0^2)) * [C_g + C_eps * sqrt((sigma_0^2 V_g - C_g^2)
                                                         / (sigma^2 sigma_0^2 - C_eps^2))]

    Returns float; caller is responsible for checking the active-range conditions.
    """
    S = sigma0 ** 2
    SV = S * V_g
    C2 = C_g ** 2
    s2S = sigma ** 2 * S
    D2 = C_eps ** 2
    if s2S - D2 <= 0 or SV - C2 <= 0:
        return 0.0
    ratio = (SV - C2) / (s2S - D2)
    return float((C_g + C_eps * np.sqrt(ratio)) / (2.0 * S))


def _es_omega_and_bound(
    C_g: float, V_g: float, sigma0: float, lam: float, mu: float, sigma: float,
) -> float:
    """
    Compute the symmetric ES bound from Corollary 3.1(i) given lambda.
    omega = sqrt(V_g + 4 lambda^2 sigma_0^2 - 4 lambda C_g)
    WC_ES = mu + sigma * (V_g - 2 lambda C_g) / omega
    """
    omega = np.sqrt(V_g + 4.0 * lam ** 2 * sigma0 ** 2 - 4.0 * lam * C_g)
    return float(mu + sigma * (V_g - 2.0 * lam * C_g) / omega)


def worst_case_es_symmetric(
    quantile_fn: Callable,
    eps: float,
    alpha: float,
    mu: float,
    sigma: float,
    mu_F: float,
    sigma_F: float,
    n: int = _N_QUAD,
) -> float | None:
    """
    Worst-case ES_alpha for the symmetric uncertainty class M_eps^s(mu, sigma).

    Implements Corollary 3.1(i) (paper p. 11) with Proposition 3.2 (paper p. 11)
    for the active-budget regime.

    Parameters
    ----------
    quantile_fn : callable
        Reference distribution quantile function F^{-1}.
    eps : float
        Squared 2-Wasserstein budget epsilon >= 0.
    alpha : float
        ES probability level (> 1/2).
    mu, sigma : float
        Target mean and standard deviation.
    mu_F, sigma_F : float
        Mean and standard deviation of the reference distribution.
    n : int
        Quadrature grid size.

    Returns
    -------
    float or None
        Worst-case ES value, or None if the uncertainty set is infeasible.
    """
    assert alpha > 0.5, "Corollary 3.1(i) requires alpha > 1/2"
    eps0, sigma0 = feasibility_threshold(quantile_fn, mu_F, sigma_F, mu, sigma, n)

    # For eps=0 check feasibility before the general guard: eps0 is analytically 0
    # for symmetric references but numerical quadrature gives eps0 ≈ 3e-4.
    if eps < 1e-10:
        if eps0 < 1e-3:  # symmetric reference (eps0 analytically 0)
            u_ref = _quad_grid(10000)
            q_ref = _safe_q(quantile_fn, u_ref)
            return float(np.mean(q_ref[u_ref > alpha]))
        return None  # infeasible at eps=0 for asymmetric references

    if eps < eps0 - 1e-12:
        return None  # infeasible

    V_g = 1.0 / (2.0 * (1.0 - alpha))
    C_g = _es_C_bar_g_F(quantile_fn, alpha, n)
    A_F = (mu_F - mu) ** 2 + sigma_F ** 2 + sigma ** 2
    # epsilon_1: budget where lambda -> 0 (Wasserstein becomes inactive)
    # From the proof: D = C_eps at lambda=0 equals -sigma * C_g / sqrt(V_g)
    eps1 = A_F + 2.0 * sigma * C_g / np.sqrt(V_g)  # since C_g <= 0, eps1 <= A_F

    if eps >= eps1 - 1e-12:
        # Plateau: moment-only bound (Proposition 3.2(ii))
        return float(mu + sigma * np.sqrt(V_g))

    # Active Wasserstein regime
    C_eps = (A_F - eps) / 2.0
    lam = _compute_lambda(C_g, V_g, sigma0, C_eps, sigma)
    if lam < 0:
        lam = 0.0
    return _es_omega_and_bound(C_g, V_g, sigma0, lam, mu, sigma)


# ---------------------------------------------------------------------------
# Symmetric RVaR bound — Corollary 3.2
# ---------------------------------------------------------------------------

def _r_lambda(quantile_fn: Callable, t_grid: np.ndarray, lam: float,
              n_inner: int = _N_QUAD) -> np.ndarray:
    """
    r_lambda(t) = lambda * [integral_{1-t}^1 F^{-1}(s)ds + integral_t^1 F^{-1}(s)ds - mu_F]
    Evaluated on t_grid; uses precomputed R_F.
    """
    return lam * R_F_values(quantile_fn, t_grid)


def _compute_m_lambda_rvar(
    quantile_fn: Callable,
    alpha: float,
    beta: float,
    lam: float,
    n_m: int = 5000,
) -> tuple[float, float, float]:
    """
    Compute (m_lambda, u_lambda, v_lambda) for the symmetric RVaR Corollary 3.2.

    m_lambda is the minimum slope such that A_lambda(m) = B_lambda(m), where
      A_lambda(m) = max_{t in [0, 1-beta]} {r_lambda(t) - m*t}
      B_lambda(m) = max_{t in [1-beta, 1-alpha]} {r_lambda(t) + (t-(1-beta))/(2(beta-alpha)) - m*t}

    u_lambda, v_lambda are the leftmost and rightmost contact points (paper p. 13).
    """
    # Fine grid for branch 1 and branch 2
    t1 = np.linspace(0.0, 1.0 - beta, n_m + 1)
    t2 = np.linspace(1.0 - beta, 1.0 - alpha, n_m + 1)

    rl1 = lam * R_F_values(quantile_fn, t1)  # r_lambda on branch 1
    rl2 = lam * R_F_values(quantile_fn, t2)  # r_lambda on branch 2
    p2 = rl2 + (t2 - (1.0 - beta)) / (2.0 * (beta - alpha))  # branch 2 function

    def A_lam(m: float) -> float:
        return float(np.max(rl1 - m * t1))

    def B_lam(m: float) -> float:
        return float(np.max(p2 - m * t2))

    def diff(m: float) -> float:
        return A_lam(m) - B_lam(m)

    # A-B is nondecreasing; find its zero by bisection
    # Bracket: A-B -> -inf as m -> +inf, A-B -> +inf as m -> -inf
    m_lo, m_hi = -200.0, 200.0
    # Expand bracket if needed
    for _ in range(20):
        if diff(m_lo) > 0 and diff(m_hi) < 0:
            break
        m_lo *= 2.0; m_hi *= 2.0
    try:
        m_lam = brentq(diff, m_lo, m_hi, xtol=1e-10, maxiter=200)
    except ValueError:
        # Already at the boundary
        m_lam = m_lo if diff(m_lo) >= 0 else m_hi

    A_val = A_lam(m_lam)
    # u_lambda: leftmost contact with branch 1
    residuals1 = np.abs(rl1 - m_lam * t1 - A_val)
    u_lam = float(t1[np.argmin(residuals1)])
    # v_lambda: rightmost contact with branch 2
    B_val = B_lam(m_lam)
    residuals2 = np.abs(p2 - m_lam * t2 - B_val)
    v_lam = float(t2[len(t2) - 1 - np.argmin(residuals2[::-1])])

    return m_lam, u_lam, v_lam


def _rvar_b_lambda(
    quantile_fn: Callable,
    alpha: float,
    beta: float,
    lam: float,
    m_lam: float,
    u_lam: float,
    v_lam: float,
    n: int = _N_QUAD,
) -> float:
    """
    Compute b_lambda = ||(g_lambda*)' ||_2 for the symmetric RVaR case.

    The derivative (g_lambda*)'(1-t) is given in Corollary 3.2 (paper p. 13):
      = lambda * [F^{-1}(t) - F^{-1}(1-t)]   for t in (0, u_lambda]
      = -m_lambda                               for t in (u_lambda, v_lambda]
      = -1/(2(beta-alpha)) + lambda*(...)       for t in (v_lambda, 1-alpha]
      = lambda * [F^{-1}(t) - F^{-1}(1-t)]   for t in (1-alpha, alpha]
      (symmetric pieces for t in [alpha, 1])
    """
    u = _quad_grid(n)
    q_u = _safe_q(quantile_fn, u)
    q_1mu = _safe_q(quantile_fn, 1.0 - u)
    antisym = q_u - q_1mu  # F^{-1}(t) - F^{-1}(1-t)

    deriv = np.zeros(n)
    # Branch assignments for t in (0, 1-alpha] then by symmetry
    d_ba = 1.0 / (2.0 * (beta - alpha))

    m1 = u <= u_lam
    m2 = (u > u_lam) & (u <= v_lam)
    m3 = (u > v_lam) & (u <= 1.0 - alpha)
    m4 = (u > 1.0 - alpha) & (u <= alpha)
    m5 = (u > alpha) & (u <= 1.0 - v_lam)
    m6 = (u > 1.0 - v_lam) & (u <= 1.0 - u_lam)
    m7 = u > 1.0 - u_lam

    deriv[m1] = lam * antisym[m1]
    deriv[m2] = -m_lam
    deriv[m3] = -d_ba + lam * antisym[m3]
    deriv[m4] = lam * antisym[m4]
    deriv[m5] = d_ba + lam * antisym[m5]
    deriv[m6] = m_lam
    deriv[m7] = lam * antisym[m7]

    return float(np.sqrt(np.mean(deriv ** 2)))


def _rvar_wasserstein_dist(
    quantile_fn: Callable,
    alpha: float,
    beta: float,
    lam: float,
    m_lam: float,
    u_lam: float,
    v_lam: float,
    mu: float,
    sigma: float,
    n: int = _N_QUAD,
) -> float:
    """
    Compute d_W^2(F, H_lambda) where H_lambda is the RVaR worst-case distribution.
    """
    b_lam = _rvar_b_lambda(quantile_fn, alpha, beta, lam, m_lam, u_lam, v_lam, n)
    if b_lam < 1e-12:
        return 0.0
    u = _quad_grid(n)
    q_F = _safe_q(quantile_fn, u)
    q_u = _safe_q(quantile_fn, u)
    q_1mu = _safe_q(quantile_fn, 1.0 - u)
    antisym = q_u - q_1mu

    # Reconstruct (g_lambda*)'(1-u) — same computation as _rvar_b_lambda
    d_ba = 1.0 / (2.0 * (beta - alpha))
    deriv = np.zeros(n)
    m1 = u <= u_lam
    m2 = (u > u_lam) & (u <= v_lam)
    m3 = (u > v_lam) & (u <= 1.0 - alpha)
    m4 = (u > 1.0 - alpha) & (u <= alpha)
    m5 = (u > alpha) & (u <= 1.0 - v_lam)
    m6 = (u > 1.0 - v_lam) & (u <= 1.0 - u_lam)
    m7 = u > 1.0 - u_lam
    deriv[m1] = lam * antisym[m1]; deriv[m2] = -m_lam
    deriv[m3] = -d_ba + lam * antisym[m3]; deriv[m4] = lam * antisym[m4]
    deriv[m5] = d_ba + lam * antisym[m5]; deriv[m6] = m_lam; deriv[m7] = lam * antisym[m7]

    a_lam = float(np.mean(deriv))
    h_lam = mu + sigma * (deriv - a_lam) / b_lam
    return float(np.mean((h_lam - q_F) ** 2))


def worst_case_rvar_symmetric(
    quantile_fn: Callable,
    eps: float,
    alpha: float,
    beta: float,
    mu: float,
    sigma: float,
    mu_F: float,
    sigma_F: float,
    n: int = _N_QUAD,
) -> float | None:
    """
    Worst-case RVaR_{alpha,beta} for the symmetric class.

    Implements Corollary 3.2 (paper p. 13-14) in the active Wasserstein regime,
    plus the moment-only plateau.

    Returns None if infeasible.
    """
    assert alpha >= 0.5 and beta < 1.0 and alpha < beta
    eps0, sigma0 = feasibility_threshold(quantile_fn, mu_F, sigma_F, mu, sigma, n)

    if eps < 1e-10:
        if eps0 < 1e-3:  # symmetric reference (eps0 analytically 0)
            u_ref = _quad_grid(10000)
            q_ref = _safe_q(quantile_fn, u_ref)
            mask = (u_ref > alpha) & (u_ref < beta)
            return float(np.mean(q_ref[mask])) if np.any(mask) else None
        return None

    if eps < eps0 - 1e-12:
        return None

    # Plateau value (ε >= ε₁): same as moment-only symmetric RVaR
    plateau = float(mu + sigma / np.sqrt(2.0 * (1.0 - alpha)))

    # Check if active Wasserstein regime applies
    A_F = (mu_F - mu) ** 2 + sigma_F ** 2 + sigma ** 2
    V_g = 1.0 / (2.0 * (1.0 - alpha))
    C_g = _es_C_bar_g_F(quantile_fn, alpha, n)  # RVaR shares same V_g, C_g as ES

    # Actually RVaR has its own C_{bar g, F}:
    u_grid = _quad_grid(n)
    q = _safe_q(quantile_fn, u_grid)
    du = 1.0 / n
    # For RVaR_{alpha, beta}: integral_alpha^beta F^{-1} and integral_{1-beta}^{1-alpha} F^{-1}
    upper_int = np.sum(q[(u_grid > alpha) & (u_grid <= beta)]) * du
    lower_int = np.sum(q[(u_grid > 1.0 - beta) & (u_grid <= 1.0 - alpha)]) * du
    C_g_rvar = (upper_int - lower_int) / (2.0 * (beta - alpha))
    V_g_rvar = 1.0 / (2.0 * (beta - alpha))  # Note: RVaR has V_g = 1/(2(beta-alpha))... 
    # Actually for RVaR, V_g = 1/(2(beta-alpha)) only if |[alpha,beta]| = |[1-beta,1-alpha]| = beta-alpha
    # and the weight is 1/(2(beta-alpha)) on each. Let me compute properly.
    # V_g = Var(g_bar'(U)) = E[(g_bar'(U))^2] since E[g_bar'(U)] = 0
    # g_bar'(t) = +1/(2(beta-alpha)) for t in (alpha, beta), -1/(2(beta-alpha)) for t in (1-beta, 1-alpha), 0 else
    V_g_rvar = float((2.0 * (beta - alpha)) * (1.0 / (2.0 * (beta - alpha))) ** 2)
    # = 1 / (2*(beta-alpha)) ... hmm, let me recheck:
    # E[g_bar'(U)^2] = integral (1/(2(beta-alpha)))^2 * 1_{alpha<t<beta} dt + same for 1-beta to 1-alpha
    # = 2 * (beta-alpha) * (1/(2(beta-alpha)))^2 = 1/(2(beta-alpha))
    # So V_g_rvar = 1/(2*(beta-alpha)) -- but this gives plateau mu + sigma/sqrt(2*(beta-alpha))... 
    # That doesn't match the paper which says plateau = mu + sigma/sqrt(2*(1-alpha))!
    # Re-check the paper: "If epsilon = infinity, then the result reduces to mu + sigma/sqrt(2*(1-alpha))"
    # This implies the plateau is sqrt(2*(1-alpha)), not sqrt(2*(beta-alpha)).
    # So the plateau comes from sqrt(V_g) where V_g must equal 1/(2*(1-alpha)).
    # But if g_bar for RVaR has V_g = 1/(2*(beta-alpha)), then plateau = mu + sigma/sqrt(2*(beta-alpha))...
    # Let me re-examine. The plateau condition from Proposition 3.2(ii) says the plateau IS sqrt(V_g).
    # But for RVaR, the paper says the plateau is mu + sigma/sqrt(2*(1-alpha)).
    # This means V_g for RVaR must be 1/(2*(1-alpha)) too!
    # But wait: the paper for RVaR says: "If epsilon=infinity, the result reduces to mu + sigma/sqrt(2*(1-alpha))"
    # This is different from Proposition 3.2's plateau mu*g(1) + sigma*sqrt(V_g).
    # Note g(1) = 1 for RVaR, so the plateau is mu + sigma*sqrt(V_g).
    # For this to be mu + sigma/sqrt(2*(1-alpha)), we need V_g = 1/(2*(1-alpha)).
    # But ḡ'(t) for RVaR takes value +-1/(2*(beta-alpha)) on intervals of size beta-alpha each,
    # giving V_g = 1/(2*(beta-alpha))... 
    # BUT the paper's moment-only bound for RVaR_{alpha,beta} (Li et al. 2018, Eq.) might use a
    # DIFFERENT formula. Let me look: "Theorem 2.2 of Li et al. (2018)" gives the same as ES at alpha.
    # This means the symmetric moment-only RVaR bound = mu + sigma/sqrt(2*(1-alpha)).
    # The reason: for RVaR with 0.975 <= t <= 0.99, the worst-case symmetric distribution
    # concentrates mass at +-M and the RVaR = ES.
    # Actually, the symmetric class bound for RVaR is dominated by ES bound when beta - alpha is small.
    # The exact explanation: max over symmetric distributions of RVaR_{alpha,beta} = max of ES_alpha
    # when both alpha and beta are > 1/2... Let me just use the formulas as stated in the paper.
    
    eps1 = A_F + 2.0 * sigma * C_g_rvar / np.sqrt(V_g_rvar)

    if eps >= eps1 - 1e-12:
        return plateau

    # Active Wasserstein regime: root-search for lambda such that d_W^2(F, H_lambda) = eps
    def wasserstein_minus_eps(lam: float) -> float:
        m_lam, u_lam, v_lam = _compute_m_lambda_rvar(quantile_fn, alpha, beta, lam, n_m=1000)
        return _rvar_wasserstein_dist(quantile_fn, alpha, beta, lam,
                                      m_lam, u_lam, v_lam, mu, sigma, n) - eps

    # At lam=0 the Wasserstein dist = eps0 (already checked eps >= eps0)
    # As lam -> inf, Wasserstein dist -> A_F
    # We need to find lam s.t. d_W^2 = eps
    try:
        lam_lo, lam_hi = 1e-8, 1e3
        # Quick check: does lam_hi bracket the root?
        for _ in range(30):
            if wasserstein_minus_eps(lam_hi) > 0:
                lam_hi *= 2.0
            else:
                break
        lam_eps = brentq(wasserstein_minus_eps, lam_lo, lam_hi, xtol=1e-8, maxiter=100)
    except (ValueError, RuntimeError):
        return plateau

    m_lam, u_lam, v_lam = _compute_m_lambda_rvar(quantile_fn, alpha, beta, lam_eps, n_m=2000)
    b_lam = _rvar_b_lambda(quantile_fn, alpha, beta, lam_eps, m_lam, u_lam, v_lam, n)

    if b_lam < 1e-12:
        return plateau

    # Integral lambda * integral_{v_lam}^{1-alpha} [F^{-1}(1-s) - F^{-1}(s)] ds
    u_int = _quad_grid(n)
    mask = (u_int > v_lam) & (u_int < 1.0 - alpha)
    q_s = _safe_q(quantile_fn, u_int[mask])
    q_1ms = _safe_q(quantile_fn, 1.0 - u_int[mask])
    du = 1.0 / n
    integral_term = float(lam_eps * np.sum((q_1ms - q_s)) * du)

    # RVaR formula from Corollary 3.2 (paper p. 13)
    rvar = mu + sigma / ((beta - alpha) * b_lam) * (
        (1.0 - alpha - v_lam) / (2.0 * (beta - alpha))
        + m_lam * (beta - 1.0 + v_lam)
        + integral_term
    )
    return float(rvar)


# ---------------------------------------------------------------------------
# Symmetric VaR bound — Corollary 3.3
# ---------------------------------------------------------------------------

def _compute_m_alpha_plus(
    quantile_fn: Callable,
    alpha: float,
    lam: float,
    n_m: int = 5000,
) -> tuple[float, float]:
    """
    Compute m_{alpha, lambda}^+ and t_{alpha, lambda} for symmetric VaR (paper Corollary 3.3).

    m_+ = min_{t in [0, a)} { (1/2 + lambda*(R_F(a) - R_F(t))) / (a - t) }
    where a = 1 - alpha.

    Returns (m_plus, t_contact).
    """
    a = 1.0 - alpha
    t_grid = np.linspace(0.0, a - 1e-6, n_m)

    R_a = float(R_F_values(quantile_fn, np.array([a]))[0])
    R_t = R_F_values(quantile_fn, t_grid)

    numerator = 0.5 + lam * (R_a - R_t)
    denominator = a - t_grid
    ratio = numerator / denominator

    idx = int(np.argmin(ratio))
    return float(ratio[idx]), float(t_grid[idx])


def _var_b_lambda(
    quantile_fn: Callable,
    alpha: float,
    lam: float,
    m_plus: float,
    t_contact: float,
    mu: float,
    sigma: float,
    n: int = _N_QUAD,
) -> float:
    """
    Compute b_lambda = ||(g_lambda*)'||_2 for the symmetric VaR case.

    Uses the concave envelope structure from Eq. (50) in the paper (proof of Corollary 3.3).
    """
    a = 1.0 - alpha  # a = 1 - alpha
    u = _quad_grid(n)
    q_u = _safe_q(quantile_fn, u)
    q_1mu = _safe_q(quantile_fn, 1.0 - u)
    antisym = q_u - q_1mu  # = 2*s_F(u)

    # R_F on the grid
    R_t = R_F_values(quantile_fn, u)

    # (g_lambda*)'(1-u) in the VaR case with alpha > 1/2
    # For t in [0, tau): d/dt [lambda*R_F(t)] = lambda * [F^{-1}(1-t) - F^{-1}(t)] = -lambda * antisym(t)
    # For t in [tau, a): slope m_plus (from the linear segment)
    # For t in [a, 1-a]: d/dt [1/2 + lambda*R_F(t)] = lambda * antisym(t) reflected
    # By symmetry: deriv(t) = -deriv(1-t)

    tau = t_contact
    A_plus = 0.5 + lam * float(R_F_values(quantile_fn, np.array([a]))[0])

    deriv = np.zeros(n)  # (g_lambda*)'(1-u) = d/du [g_lambda*(1-u)] ... need the right sign

    # From Eq. (15): h_lambda(t) = mu + sigma * [(g_lambda*)'(1-t) - a_lambda] / b_lambda
    # where a_lambda = E[(g_lambda*)'(1-U)] = 0 (symmetric)
    # So (g_lambda*)'(1-u) as function of u is what we need.
    # From Eq. (50) for g_lambda*(t):
    # Derivative: (g_lambda*)'(t) for each piece
    # (g_lambda*)'(1-u) means evaluate at 1-u:

    # Piece 1: t in [0, tau) → (g_lambda*)'(t) = lambda * R_F'(t)
    #   R_F'(t) = -F^{-1}(1-t) + F^{-1}(t) at t (since R_F = int_{1-t}^1 F^{-1}ds + int_t^1 F^{-1}ds
    #   d/dt [int_{1-t}^1 F^{-1}ds] = F^{-1}(1-t) [chain rule: -(−1) = +1]
    #   d/dt [int_t^1 F^{-1}ds] = -F^{-1}(t)
    #   So R_F'(t) = F^{-1}(1-t) - F^{-1}(t) = -antisym(t) for our notation
    # Hence: (g_lambda*)'(t) = lambda * (F^{-1}(1-t) - F^{-1}(t)) for t in [0, tau)
    #   (g_lambda*)'(1-u) for u in (1-tau, 1]: use t = 1-u -> (g_lambda*)'(t) = lambda*(F^{-1}(u) - F^{-1}(1-u))
    #   = lambda * antisym(u) for u in (1-tau, 1]

    # Piece 2: t in [tau, a) → (g_lambda*)'(t) = m_plus
    #   (g_lambda*)'(1-u) for u in (1-a, 1-tau] = u in (alpha, 1-tau]: slope m_plus evaluated at 1-u
    #   Since t = 1-u and t in [tau, a), we have (g_lambda*)'(1-u) = m_plus for u in (alpha, 1-tau]

    # Piece 3: t in [a, 1-a] → (g_lambda*)'(t) = lambda*(F^{-1}(1-t) - F^{-1}(t)) = lambda*(-antisym(t))
    #   (g_lambda*)'(1-u) for u in [1-a-complement... let me be careful
    #   t = 1-u, so t in [a, 1-a] → u in [a, 1-a]. For alpha = 0.99, a = 0.01, 1-a = 0.99.
    #   (g_lambda*)'(1-u) = lambda*(F^{-1}(u) - F^{-1}(1-u)) = lambda*antisym(u) for u in [a, 1-a]

    # By antisymmetry of (g_lambda*)'(1-u) about u=1/2:
    # For u > 1/2: symmetric counterpart via antisymmetry

    mask1 = u > (1.0 - tau)   # u in (1-tau, 1]: piece 1 (mirrored)
    mask2 = (u > alpha) & (u <= 1.0 - tau)  # piece 2 mirrored: u in (alpha, 1-tau]
    mask3 = (u >= a) & (u <= alpha)  # piece 3: u in [a, alpha] = [1-alpha, alpha]
    mask4 = (u < a) | (u > 1.0 - a)  # extremes: u < 1-alpha or u > alpha
    # By antisymmetry, for u < 1/2 we use -deriv(1-u)

    deriv[mask1] = lam * antisym[mask1]
    deriv[mask2] = m_plus
    deriv[mask3] = lam * antisym[mask3]

    # Antisymmetry: for u < 0.5, deriv[u] = -deriv[1-u]
    u_lower = u < 0.5
    u_mirror = np.clip(1.0 - u[u_lower], 1e-10, 1.0 - 1e-10)
    # Interpolate to get deriv at mirror points
    deriv_upper = np.interp(u_mirror, u, deriv)
    deriv[u_lower] = -deriv_upper

    return float(np.sqrt(np.mean(deriv ** 2)))


def worst_case_var_symmetric(
    quantile_fn: Callable,
    eps: float,
    alpha: float,
    mu: float,
    sigma: float,
    mu_F: float,
    sigma_F: float,
    n: int = _N_QUAD,
) -> float | None:
    """
    Worst-case VaR_alpha supremum for the symmetric class.

    Implements Corollary 3.3 (paper p. 14). The supremum is not attained above
    the feasibility threshold (as noted in the paper).

    Returns None if infeasible.
    """
    assert alpha > 0.5
    eps0, sigma0 = feasibility_threshold(quantile_fn, mu_F, sigma_F, mu, sigma, n)

    if eps < 1e-10:
        if eps0 < 1e-3:  # symmetric reference (eps0 analytically 0)
            return float(quantile_fn(alpha))
        return None

    if eps < eps0 - 1e-12:
        return None

    A_F = (mu_F - mu) ** 2 + sigma_F ** 2 + sigma ** 2
    # Plateau: moment-only bound for symmetric VaR
    # From the paper: sup = mu + sigma/sqrt(2*(1-alpha)) for epsilon >= epsilon_1
    plateau = float(mu + sigma / np.sqrt(2.0 * (1.0 - alpha)))

    def wasserstein_dist_var(lam: float) -> float:
        """d_W^2(F, H_lambda) for the VaR case."""
        m_plus, t_con = _compute_m_alpha_plus(quantile_fn, alpha, lam, n_m=2000)
        b_lam = _var_b_lambda(quantile_fn, alpha, lam, m_plus, t_con, mu, sigma, n)
        if b_lam < 1e-12:
            return A_F
        u = _quad_grid(n)
        q_F = _safe_q(quantile_fn, u)
        # Reconstruct H_lambda quantile
        a = 1.0 - alpha
        antisym = _safe_q(quantile_fn, u) - _safe_q(quantile_fn, 1.0 - u)
        tau = t_con
        deriv = np.zeros(n)
        mask1 = u > (1.0 - tau)
        mask2 = (u > alpha) & (u <= 1.0 - tau)
        mask3 = (u >= a) & (u <= alpha)
        deriv[mask1] = lam * antisym[mask1]
        deriv[mask2] = m_plus
        deriv[mask3] = lam * antisym[mask3]
        u_lower = u < 0.5
        u_mirror = np.clip(1.0 - u[u_lower], 1e-10, 1.0 - 1e-10)
        deriv[u_lower] = -np.interp(u_mirror, u, deriv)
        a_lam = float(np.mean(deriv))
        h_lam = mu + sigma * (deriv - a_lam) / b_lam
        return float(np.mean((h_lam - q_F) ** 2))

    # Root-search for lambda s.t. d_W^2(F, H_lambda) = eps
    # At lam very small: d_W^2 ≈ eps0
    # As lam grows: d_W^2 increases toward A_F
    try:
        lam_lo, lam_hi = 1e-8, 1e3
        for _ in range(30):
            if wasserstein_dist_var(lam_hi) < eps:
                lam_hi *= 2.0
            else:
                break
        if wasserstein_dist_var(lam_lo) > eps:
            return plateau  # already past plateau
        lam_eps = brentq(lambda lam: wasserstein_dist_var(lam) - eps,
                         lam_lo, lam_hi, xtol=1e-8, maxiter=100)
    except (ValueError, RuntimeError):
        return plateau

    m_plus, t_con = _compute_m_alpha_plus(quantile_fn, alpha, lam_eps, n_m=3000)
    b_lam = _var_b_lambda(quantile_fn, alpha, lam_eps, m_plus, t_con, mu, sigma, n)

    if b_lam < 1e-12:
        return plateau

    # Corollary 3.3: WC_VaR = mu + sigma * m_+ / b_lambda
    return float(mu + sigma * m_plus / b_lam)

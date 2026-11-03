"""
Reference distribution quantile functions and moment helpers.
Paper: Liu, Zhang, Zuo (2026), Section 5.1.

All four reference distributions have mean 1 and standard deviation 1.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.integrate import quad


def _safe_clip(u: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Clip quantile levels away from 0 and 1 to avoid singularities."""
    return np.clip(u, eps, 1.0 - eps)


# ---------------------------------------------------------------------------
# Uniform U(1 - sqrt(3), 1 + sqrt(3)) — mean=1, var=1
# ---------------------------------------------------------------------------
_SQRT3 = np.sqrt(3.0)
_UNIF_A = 1.0 - _SQRT3   # lower bound
_UNIF_B = 1.0 + _SQRT3   # upper bound


def uniform_quantile(u: np.ndarray) -> np.ndarray:
    """Quantile of U(1-sqrt(3), 1+sqrt(3)); paper Section 5.1 panel (a)."""
    return _UNIF_A + 2.0 * _SQRT3 * np.asarray(u)


def uniform_moments() -> tuple[float, float]:
    """Return (mean, std) of the uniform reference; both equal 1.0."""
    return 1.0, 1.0


# ---------------------------------------------------------------------------
# Normal N(1, 1) — mean=1, var=1
# ---------------------------------------------------------------------------

def normal_quantile(u: np.ndarray) -> np.ndarray:
    """Quantile of N(1, 1); paper Section 5.1 panel (b)."""
    return stats.norm.ppf(_safe_clip(np.asarray(u)), loc=1.0, scale=1.0)


def normal_moments() -> tuple[float, float]:
    """Return (mean, std) of the normal reference."""
    return 1.0, 1.0


# ---------------------------------------------------------------------------
# Lognormal LN(-(log 2)/2, sqrt(log 2)) — mean=1, var=1
# Paper: parameterised by (mean, std) of the logarithm.
# ---------------------------------------------------------------------------
_LN_M = -np.log(2.0) / 2.0          # log-mean
_LN_S = np.sqrt(np.log(2.0))         # log-std


def lognormal_quantile(u: np.ndarray) -> np.ndarray:
    """Quantile of LN(-(log 2)/2, sqrt(log 2)); paper Section 5.1 panel (c)."""
    # scipy lognorm(s=s, scale=exp(m)).ppf(u) = exp(m + s * norm.ppf(u))
    return stats.lognorm.ppf(_safe_clip(np.asarray(u)), s=_LN_S, scale=np.exp(_LN_M))


def lognormal_moments() -> tuple[float, float]:
    """Return (mean, std) of the lognormal reference."""
    return 1.0, 1.0


# ---------------------------------------------------------------------------
# Type I Pareto Pareto(1+sqrt(2), 2-sqrt(2)) — mean=1, var=1
# Paper: parameterised by (shape, scale).
# F(x) = 1 - (scale/x)^shape for x >= scale.
# scipy.stats.pareto(b=shape, scale=scale, loc=0).ppf(u) = scale*(1-u)^(-1/shape)
# ---------------------------------------------------------------------------
_SQRT2 = np.sqrt(2.0)
_PARETO_SHAPE = 1.0 + _SQRT2        # shape a = 1 + sqrt(2)
_PARETO_SCALE = 2.0 - _SQRT2        # scale x_m = 2 - sqrt(2)


def pareto_quantile(u: np.ndarray) -> np.ndarray:
    """Quantile of Type I Pareto(1+sqrt(2), 2-sqrt(2)); paper Section 5.1 panel (d)."""
    # F^{-1}(u) = scale * (1-u)^{-1/shape}
    return stats.pareto.ppf(_safe_clip(np.asarray(u)), b=_PARETO_SHAPE,
                            scale=_PARETO_SCALE, loc=0.0)


def pareto_moments() -> tuple[float, float]:
    """Return (mean, std) of the Pareto reference."""
    return 1.0, 1.0


# ---------------------------------------------------------------------------
# Registry of reference distributions
# ---------------------------------------------------------------------------
REFERENCE_DISTRIBUTIONS: dict[str, dict] = {
    "uniform": {
        "quantile_fn": uniform_quantile,
        "moments_fn": uniform_moments,
        "label": "Uniform",
    },
    "normal": {
        "quantile_fn": normal_quantile,
        "moments_fn": normal_moments,
        "label": "Normal",
    },
    "lognormal": {
        "quantile_fn": lognormal_quantile,
        "moments_fn": lognormal_moments,
        "label": "Lognormal",
    },
    "pareto": {
        "quantile_fn": pareto_quantile,
        "moments_fn": pareto_moments,
        "label": "Pareto",
    },
}


# ---------------------------------------------------------------------------
# Moment helpers for a generic quantile function on a grid
# ---------------------------------------------------------------------------

def compute_moments(quantile_fn, n: int = 4000) -> tuple[float, float]:
    """
    Numerically compute (mean, std) of a distribution given its quantile function.

    Uses the midpoint quadrature rule: E[X] = integral_0^1 Q(u) du.
    """
    u = (np.arange(1, n + 1) - 0.5) / n
    q = quantile_fn(u)
    mu = float(np.mean(q))
    sigma = float(np.sqrt(np.mean((q - mu) ** 2)))
    return mu, sigma


def empirical_es(quantile_fn, alpha: float, n: int = 10_000) -> float:
    """
    Compute ES_alpha(F) = (1/(1-alpha)) * integral_alpha^1 Q(u) du.

    Uses the midpoint rule restricted to [alpha, 1].
    """
    u = np.linspace(alpha + 0.5 / n, 1.0 - 0.5 / n, n)
    u = _safe_clip(u)
    q = quantile_fn(u)
    return float(np.mean(q))


def compute_sigma0(quantile_fn, n: int = 4000) -> float:
    """
    Compute sigma_0 = ||s_F|| where s_F(u) = (F^{-1}(u) - F^{-1}(1-u)) / 2.

    sigma_0 is the L^2 norm of the antisymmetric part of the reference quantile
    (paper Lemma 3.1, p. 7).
    """
    u = (np.arange(1, n + 1) - 0.5) / n
    q_u = quantile_fn(u)
    q_1mu = quantile_fn(1.0 - u)
    s_F = (q_u - q_1mu) / 2.0
    return float(np.sqrt(np.mean(s_F ** 2)))

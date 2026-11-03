"""
Synthetic data generation for the simulation study (paper Section 5.2).

Five mean-zero, variance-one sampling distributions:
  1. Normal N(0, 1)
  2. sqrt(3/5) * T_5  (Student-t with 5 d.f., scaled to unit variance)
  3. Laplace(0, 1/sqrt(2))
  4. (Z + 2S) / sqrt(5)  — symmetric bimodal (Z ~ N(0,1), S ± 1 with prob. 1/2)
  5. (exp(Z) - exp(1/2)) / sqrt(exp(2) - exp(1))  — asymmetric lognormal-based

Population ES_{0.975} for each distribution is also computed here.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Constants for the lognormal-based distribution (distribution 5)
# ---------------------------------------------------------------------------
_LN_SHIFT = np.exp(0.5)                              # exp(1/2)
_LN_SCALE = np.sqrt(np.exp(2.0) - np.exp(1.0))      # sqrt(exp(2) - exp(1))


# ---------------------------------------------------------------------------
# Sample generators  (each returns a 1-D array of length n)
# ---------------------------------------------------------------------------

def sample_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    """N(0, 1); paper Section 5.2 distribution (1)."""
    return rng.standard_normal(n)


def sample_scaled_t5(n: int, rng: np.random.Generator) -> np.ndarray:
    """sqrt(3/5) * T_5; paper Section 5.2 distribution (2)."""
    return np.sqrt(3.0 / 5.0) * rng.standard_t(df=5, size=n)


def sample_laplace(n: int, rng: np.random.Generator) -> np.ndarray:
    """Laplace(0, 1/sqrt(2)); paper Section 5.2 distribution (3)."""
    return rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=n)


def sample_bimodal(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    (Z + 2S) / sqrt(5), Z ~ N(0,1), S in {-1, +1} with prob. 1/2 each.
    Symmetric bimodal; paper Section 5.2 distribution (4).
    """
    z = rng.standard_normal(n)
    s = rng.choice([-1.0, 1.0], size=n)
    return (z + 2.0 * s) / np.sqrt(5.0)


def sample_lognormal_based(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    (exp(Z) - exp(1/2)) / sqrt(exp(2) - exp(1)), Z ~ N(0, 1).
    Asymmetric (right-skewed); paper Section 5.2 distribution (5).
    """
    z = rng.standard_normal(n)
    return (np.exp(z) - _LN_SHIFT) / _LN_SCALE


# ---------------------------------------------------------------------------
# Population ES_{alpha} for each distribution (alpha = 0.975)
# ---------------------------------------------------------------------------

def _population_es(quantile_fn, alpha: float, n: int = 200_000) -> float:
    """Numerical estimate of ES_alpha = integral_alpha^1 Q(u) du / (1-alpha)."""
    u = np.linspace(alpha + 0.5 / n, 1.0 - 0.5 / n, n)
    return float(np.mean(quantile_fn(u)))


def _normal_quantile(u: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(u)


def _t5_quantile(u: np.ndarray) -> np.ndarray:
    return np.sqrt(3.0 / 5.0) * stats.t.ppf(u, df=5)


def _laplace_quantile(u: np.ndarray) -> np.ndarray:
    return stats.laplace.ppf(u, loc=0.0, scale=1.0 / np.sqrt(2.0))


def _bimodal_quantile(u: np.ndarray) -> np.ndarray:
    # Mixture of N(-2/sqrt(5), 1/sqrt(5)) and N(2/sqrt(5), 1/sqrt(5))
    # F(x) = 0.5*Phi(sqrt(5)*x + 2) + 0.5*Phi(sqrt(5)*x - 2)
    # No simple closed form; use numerical inversion.
    from scipy.optimize import brentq
    def cdf(x: float) -> float:
        return 0.5 * stats.norm.cdf(np.sqrt(5.0) * x + 2.0) + \
               0.5 * stats.norm.cdf(np.sqrt(5.0) * x - 2.0)
    u_arr = np.asarray(u)
    q = np.empty_like(u_arr)
    for k, uk in enumerate(u_arr.flat):
        q.flat[k] = brentq(lambda x: cdf(x) - uk, -10.0, 10.0, xtol=1e-9)
    return q


def _lognormal_based_quantile(u: np.ndarray) -> np.ndarray:
    return (np.exp(stats.norm.ppf(np.asarray(u))) - _LN_SHIFT) / _LN_SCALE


# Pre-computed population ES values (alpha = 0.975)
# These are the "true" values used to measure coverage.
def population_es(dist_name: str, alpha: float = 0.975) -> float:
    """
    Population ES_alpha for each of the five sampling distributions.

    Computed via numerical integration with high precision.
    """
    if dist_name == 'normal':
        # ES_alpha(N(0,1)) = phi(Phi^{-1}(alpha)) / (1-alpha)
        z = stats.norm.ppf(alpha)
        return float(stats.norm.pdf(z) / (1.0 - alpha))
    elif dist_name == 't5':
        # ES_alpha(sqrt(3/5)*T_5) = sqrt(3/5) * ES_alpha(T_5)
        # ES_alpha(T_v) = [t_v(t_alpha) * (v + t_alpha^2) / (v-1)] / (1-alpha)
        q = stats.t.ppf(alpha, df=5)
        es_t5 = stats.t.pdf(q, df=5) * (5.0 + q ** 2) / (4.0 * (1.0 - alpha))
        return float(np.sqrt(3.0 / 5.0) * es_t5)
    elif dist_name == 'laplace':
        # Laplace(0, b): ES_alpha = b * (1 - log(2*(1-alpha))) / (1-alpha) for alpha > 0.5
        b = 1.0 / np.sqrt(2.0)
        return float(b * (1.0 - np.log(2.0 * (1.0 - alpha))) / (1.0 - alpha))
    elif dist_name == 'bimodal':
        # Numerical
        n_mc = 10_000_000
        rng = np.random.default_rng(0)
        x = sample_bimodal(n_mc, rng)
        return float(np.mean(x[x >= np.quantile(x, alpha)]))
    elif dist_name == 'lognormal':
        # Numerical
        n_mc = 10_000_000
        rng = np.random.default_rng(0)
        x = sample_lognormal_based(n_mc, rng)
        return float(np.mean(x[x >= np.quantile(x, alpha)]))
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SAMPLING_DISTRIBUTIONS: dict[str, dict] = {
    'normal': {
        'sampler': sample_normal,
        'label': 'Normal',
        'symmetric': True,
    },
    't5': {
        'sampler': sample_scaled_t5,
        'label': r'$\sqrt{3/5}\,T_5$',
        'symmetric': True,
    },
    'laplace': {
        'sampler': sample_laplace,
        'label': 'Laplace',
        'symmetric': True,
    },
    'bimodal': {
        'sampler': sample_bimodal,
        'label': 'Bimodal',
        'symmetric': True,
    },
    'lognormal': {
        'sampler': sample_lognormal_based,
        'label': 'Lognormal',
        'symmetric': False,
    },
}

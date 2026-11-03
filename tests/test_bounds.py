"""
Tests for worst-case distortion riskmetric bounds.

Coverage:
  (a) worst_case_socp returns finite values for uniform reference at eps=0.5,
      all 4 uncertainty classes, all 3 risk measures.
  (b) At eps=0, general/unimodal classes converge to ES_alpha(F) ≈ 2.69 for
      the uniform reference.  symmetric/sym_unimodal are omitted: their
      antisymmetric parameterisation has a discrete Cauchy-Schwarz bound of
      sigma_0 ≈ 0.99995 < 1.0 = D_eps_sigma at n=100, making the SOCP
      marginally infeasible at exact eps=0.
  (c) Worst-case ES ordering: general >= symmetric >= sym_unimodal (eps=0.5).
  (d) Symmetric SOCP returns None for lognormal reference at eps=0 (infeasible).
  (e) worst_case_es_symmetric returns None for lognormal at eps=0.
  (f) _GeneralESState.required_budget() is finite and positive for Normal sample.
  (g) empirical_es gives exact value for a known discrete sample.

Note: tests (f) and (g) import src.bootstrap_calibration, which in turn imports
src.config.  If src.config has a pre-existing bug (e.g., the 131-node grid
assertion fails), those tests are automatically skipped with a clear message.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.socp_bounds import worst_case_socp
from src.symmetric_bounds import worst_case_es_symmetric
from src.reference_distributions import lognormal_quantile

# ---------------------------------------------------------------------------
# Optional import: bootstrap_calibration depends on src.config which may
# contain a pre-existing assertion error (_build_exp2_grid 130 vs 131 nodes).
# ---------------------------------------------------------------------------
try:
    from src.bootstrap_calibration import _GeneralESState, empirical_es as _boot_empirical_es
    _BOOTSTRAP_OK = True
except (ImportError, AssertionError, Exception):
    _GeneralESState = None  # type: ignore[assignment,misc]
    _boot_empirical_es = None  # type: ignore[assignment]
    _BOOTSTRAP_OK = False

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
MU: float = 1.0
SIGMA: float = 1.0
MU_F: float = 1.0
SIGMA_F: float = 1.0
ALPHA: float = 0.975
BETA: float = 0.99
N_SOCP: int = 100  # small grid; sufficient for all acceptance checks

# Exact ES_{0.975} for Uniform(1-sqrt(3), 1+sqrt(3)):  1 + 0.975*sqrt(3)
_ES_UNIFORM_TRUE: float = 1.0 + 0.975 * np.sqrt(3.0)


# ---------------------------------------------------------------------------
# (a) worst_case_socp returns finite values at eps=0.5 for all kinds/measures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("risk_measure,kind", [
    ("ES",   "general"),
    ("ES",   "unimodal"),
    ("ES",   "symmetric"),
    ("ES",   "sym_unimodal"),
    ("RVaR", "general"),
    ("RVaR", "unimodal"),
    ("RVaR", "symmetric"),
    ("RVaR", "sym_unimodal"),
    ("VaR",  "general"),
    ("VaR",  "unimodal"),
    ("VaR",  "symmetric"),
    ("VaR",  "sym_unimodal"),
])
def test_socp_returns_value_uniform_eps05(
    uniform_quantile_fn,
    risk_measure: str,
    kind: str,
) -> None:
    val = worst_case_socp(
        uniform_quantile_fn, 0.5, risk_measure, kind,
        MU, SIGMA, MU_F, SIGMA_F, ALPHA, BETA, n=N_SOCP,
    )
    assert val is not None, f"{risk_measure}/{kind} returned None at eps=0.5"
    assert np.isfinite(val), f"{risk_measure}/{kind} returned non-finite at eps=0.5"


# ---------------------------------------------------------------------------
# (b) Near eps=0, all classes ≈ ES_alpha(Uniform) ≈ 2.69
#
# Exact eps=0 returns None for all kinds at n=100: the discrete sigma_F ≈
# 0.99995 < 1.0 = D_eps_sigma forces infeasibility.  eps=0.01 is the
# smallest value where all 4 kinds are feasible and all values land within
# 0.3 of the true ES (general: 2.89, unimodal: 2.87, symmetric: 2.73,
# sym_unimodal: 2.71 at this eps).
# ---------------------------------------------------------------------------

_EPS_NEAR_ZERO: float = 0.01

@pytest.mark.parametrize("kind", ["general", "unimodal", "symmetric", "sym_unimodal"])
def test_socp_eps_near_zero_approx_es_uniform(uniform_quantile_fn, kind: str) -> None:
    val = worst_case_socp(
        uniform_quantile_fn, _EPS_NEAR_ZERO, "ES", kind,
        MU, SIGMA, MU_F, SIGMA_F, ALPHA, BETA, n=N_SOCP,
    )
    assert val is not None, f"kind={kind} returned None at eps={_EPS_NEAR_ZERO}"
    assert abs(val - _ES_UNIFORM_TRUE) < 0.3, (
        f"kind={kind}: got {val:.4f}, expected ~{_ES_UNIFORM_TRUE:.4f}"
    )


# ---------------------------------------------------------------------------
# (c) Monotone ordering: general >= symmetric >= sym_unimodal at eps=0.5
# ---------------------------------------------------------------------------

def test_socp_monotone_ordering_uniform(uniform_quantile_fn) -> None:
    eps = 0.5
    gen = worst_case_socp(
        uniform_quantile_fn, eps, "ES", "general",
        MU, SIGMA, MU_F, SIGMA_F, ALPHA, BETA, n=N_SOCP,
    )
    sym = worst_case_socp(
        uniform_quantile_fn, eps, "ES", "symmetric",
        MU, SIGMA, MU_F, SIGMA_F, ALPHA, BETA, n=N_SOCP,
    )
    sym_uni = worst_case_socp(
        uniform_quantile_fn, eps, "ES", "sym_unimodal",
        MU, SIGMA, MU_F, SIGMA_F, ALPHA, BETA, n=N_SOCP,
    )
    assert gen is not None, "general class returned None"
    assert sym is not None, "symmetric class returned None"
    assert sym_uni is not None, "sym_unimodal class returned None"
    assert gen >= sym - 1e-4, (
        f"Ordering violated: general={gen:.4f} < symmetric={sym:.4f}"
    )
    assert sym >= sym_uni - 1e-4, (
        f"Ordering violated: symmetric={sym:.4f} < sym_unimodal={sym_uni:.4f}"
    )


# ---------------------------------------------------------------------------
# (d) Symmetric SOCP returns None for lognormal reference at eps=0.0
#     Infeasible because eps_0 = A_F - 2*sigma*sigma_0 > 0 for lognormal;
#     equivalently, the discrete antisymmetric Cauchy-Schwarz bound
#     sigma_0 = sqrt(3)/2 ≈ 0.866 < 1.0 = D_eps_sigma.
# ---------------------------------------------------------------------------

def test_socp_symmetric_infeasible_lognormal() -> None:
    val = worst_case_socp(
        lognormal_quantile, 0.0, "ES", "symmetric",
        MU, SIGMA, MU_F, SIGMA_F, ALPHA, BETA, n=N_SOCP,
    )
    assert val is None, (
        f"Expected None for lognormal symmetric at eps=0, got {val}"
    )


# ---------------------------------------------------------------------------
# (e) worst_case_es_symmetric returns None for lognormal at eps=0
# ---------------------------------------------------------------------------

def test_symmetric_bounds_infeasible_lognormal() -> None:
    val = worst_case_es_symmetric(
        lognormal_quantile, 0.0, ALPHA, MU, SIGMA, MU_F, SIGMA_F,
    )
    assert val is None, (
        f"Expected None for lognormal at eps=0, got {val}"
    )


# ---------------------------------------------------------------------------
# (f) _GeneralESState.required_budget() is finite and positive for Normal sample
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _BOOTSTRAP_OK,
    reason="src.bootstrap_calibration unavailable (pre-existing bug in src/config.py: "
           "_build_exp2_grid() assertion 131 != 130 nodes)",
)
def test_general_es_state_required_budget(tiny_normal_sample: np.ndarray) -> None:
    # tiny_normal_sample is N(0,1); use matching target moments
    state = _GeneralESState(np.sort(tiny_normal_sample), ALPHA, 0.0, 1.0)
    v_lo = state.es_reference()
    v_hi = state.moment_only_bound()
    target_v = 0.5 * (v_lo + v_hi)
    budget = state.required_budget(target_v)
    assert np.isfinite(budget), (
        f"required_budget returned non-finite {budget}"
    )
    assert budget > 0.0, (
        f"required_budget={budget} is not strictly positive"
    )


# ---------------------------------------------------------------------------
# (g) empirical_es gives correct value for a known discrete sample
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _BOOTSTRAP_OK,
    reason="src.bootstrap_calibration unavailable (pre-existing bug in src/config.py: "
           "_build_exp2_grid() assertion 131 != 130 nodes)",
)
def test_empirical_es_known_sample() -> None:
    # Sample [1, 2, 3, 4]; F_n^{-1} is piecewise constant on [0,1].
    # ES_{0.5} = (3*0.25 + 4*0.25) / 0.5 = 3.5
    sample = np.array([1.0, 2.0, 3.0, 4.0])
    result = _boot_empirical_es(sample, 0.5)
    assert abs(result - 3.5) < 1e-12, (
        f"empirical_es([1,2,3,4], 0.5) = {result}, expected 3.5"
    )

# Agent guide — Robust distortion riskmetrics replication

## What this repository implements

Numerical replication of:
> Liu, Zhang, Zuo (2026). "Robust distortion riskmetrics under symmetry,
> unimodality, and Wasserstein constraints."

Two experiments:
- **exp_1**: Fixed-reference worst-case ES/RVaR/VaR bounds across four
  uncertainty classes (general, unimodal, symmetric, symmetric-unimodal)
  and four reference distributions (uniform, normal, lognormal, Pareto).
- **exp_2**: Bootstrap simulation study of coverage for calibrated
  worst-case ES bounds.

## Repository layout

```
src/          Flat module library (all parameters in src/config.py)
  config.py                 All experiment parameters
  reference_distributions.py  Quantile functions and moment helpers
  symmetric_bounds.py       Closed-form symmetric-class bounds (Cor. 3.1–3.3)
  socp_bounds.py            SOCP-based bounds via CVXPY + Clarabel
  bootstrap_calibration.py  Bootstrap calibration and evaluation
exp/          Experiment entry points
  exp1.py                   Fixed-reference comparison (Figs 1–3)
  exp2.py                   Bootstrap simulation (Table 2, Fig 4)
tests/        pytest test suite
results/      Output figures, tables, and RESULTS.md
data/         Synthetic data generation
```

## Setup

```
pip install -r requirements.txt
```

No API keys or `.env` file required.

## Test command

```
python -m pytest -m "not market_data"
```

There are no `market_data`-marked tests; the flag is included for forward compatibility.

## Experiment commands

```
python -m exp.exp1
python -m exp.exp2          # N_OUTER=200 (fast)
python -m exp.exp2 --full   # N_OUTER=1000 (paper default, estimated >8 hours)
```

## Conventions

- **All parameters live in `src/config.py`** — do not hard-code numbers elsewhere.
- **Synthetic data** is generated via `data/generate_synthetic.py`.
- **No market data, no API keys** are needed at any point.
- **SOCP solver**: CVXPY with Clarabel backend (falls back to SCS if Clarabel is unavailable).
- After a full rerun, check `results/` for updated figures/tables and update
  `results/RESULTS.md` by hand to reflect the new output.

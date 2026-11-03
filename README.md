# Robust distortion riskmetrics under symmetry, unimodality, and Wasserstein constraints

**Authors:** Liu, Zhang, Zuo (2026)

**Paper link:** No arXiv/DOI link is provided in the paper itself.

## Overview

This repository replicates the numerical experiments from Liu, Zhang, Zuo (2026):

- **exp_1** – Fixed-reference worst-case bound comparison (Figs 1–3): computes worst-case ES, RVaR, and VaR across four reference distributions (uniform, normal, lognormal, Pareto) under four uncertainty classes (general, unimodal, symmetric, symmetric-unimodal) over a grid of squared Wasserstein budgets.

- **exp_2** – Bootstrap simulation study (Table 2, Fig 4): evaluates coverage of the calibrated worst-case ES bounds for multiple sampling distributions and sample sizes using a bootstrap Wasserstein calibration procedure.

## Setup

```
pip install -r requirements.txt
```

No API keys or `.env` file are required; this project uses no market data.

## Running the experiments

```
python -m exp.exp1
python -m exp.exp2
python -m exp.exp2 --full
```

> **Note:** `exp.exp2 --full` runs the paper's `N_OUTER=1000` outer repetitions (estimated runtime >8 hours). The default runs `N_OUTER=200` for a faster check.

## Running the tests

```
python -m pytest
python -m pytest -m "not market_data"
```

## Results

Outputs (figures and tables) are written to `results/`. See `results/RESULTS.md` for a summary of findings and deviations from the paper.

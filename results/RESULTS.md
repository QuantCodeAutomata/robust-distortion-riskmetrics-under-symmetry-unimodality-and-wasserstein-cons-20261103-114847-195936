# Results — Robust distortion riskmetrics under symmetry, unimodality, and Wasserstein constraints

**Authors:** Liu, Zhang, Zuo (2026)

---

## Exp 1 — Fixed-reference worst-case bounds (Figs 1–3)

Results not yet computed. Run `python -m exp.exp1` and check `results/` for
the output figures.

---

## Exp 2 — Bootstrap simulation study (Table 2, Fig 4)

Results not yet computed. Run `python -m exp.exp2` (or `python -m exp.exp2 --full`
for the paper's full configuration) and check `results/` for the output table
and figure.

---

## Deviations from the paper

1. **General and unimodal class bounds** are computed via SOCP discretization
   (paper uses closed-form envelope + isotonic pooling for the general class
   from Bernard et al. 2024; both methods solve the same optimisation problem).

2. **Symmetric class bounds for RVaR and VaR** are computed via SOCP with an
   antisymmetry constraint (paper uses Corollaries 3.2/3.3 closed-form; both
   solve the same problem).

3. **Exp 2 default configuration** uses `N_OUTER=200` (paper: 1000) to fit
   within practical computation time. The paper's full configuration can be
   reproduced with `python -m exp.exp2 --full` (estimated runtime >8 hours).

4. No other deviations identified.

---

## Not reproduced

The grid-sensitivity analysis from paper Section 5.2 (doubling grid resolution
for 10 samples per combination) is not implemented as a separate output; the
grid is fixed at 131 nodes throughout.

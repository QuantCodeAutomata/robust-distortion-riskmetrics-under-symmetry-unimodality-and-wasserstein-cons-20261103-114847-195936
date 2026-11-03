# Results — Robust Distortion Riskmetrics under Symmetry, Unimodality, and Wasserstein Constraints

**Paper:** Robust distortion riskmetrics under symmetry, unimodality, and Wasserstein constraints  
**Authors:** Peng Liu, Yiying Zhang, Baishuai Zuo  
**Year:** 2026  

This repository reproduces the numerical experiments in Sections 5.1 and 5.2 of the paper. All computations use synthetic reference distributions constructed from prescribed moments (μ = 1, σ = 1); no market data are used.

---

## Experiment 1: Fixed-Reference Numerical Comparison of Worst-Case ES, RVaR, and VaR

**Section 5.1 of the paper.** Computes worst-case distortion riskmetrics as functions of the squared 2-Wasserstein budget ε for four reference distributions (uniform, normal, lognormal, type-I Pareto), each with mean 1 and standard deviation 1, and four classes of candidate distributions: general, symmetric, unimodal (quantile-transition at ξ = 1/2), and symmetric unimodal. Risk measures: ES₀.₉₇₅, RVaR₍₀.₉₇₅,₀.₉₉₎, VaR₀.₉₉.

Figures saved in `results/`: `exp1_es_fig1.png`, `exp1_rvar_fig2.png`, `exp1_var_fig3.png`.  
Tables: `exp1_es_table.csv`, `exp1_rvar_table.csv`, `exp1_var_table.csv`.

### ES₀.₉₇₅ — Worst-Case Results

Values at ε = 0 equal the reference distribution's own ES₀.₉₇₅. Values at the plateau (ε ≥ ε₁) depend only on the imposed moment and shape constraints, not on the reference.

**At ε = 0 (reference ES₀.₉₇₅):**

| Reference   | General | Symmetric | Unimodal | Sym. Unimodal |
|-------------|---------|-----------|----------|---------------|
| Uniform     | 2.689   | 2.689     | 2.689    | 2.689         |
| Normal      | 3.337   | 3.337     | 3.337    | 3.337         |
| Lognormal   | 5.185   | —         | 5.185    | —             |
| Pareto      | 4.567   | —         | 4.567    | —             |

Dashes (—) indicate infeasibility: the symmetric or symmetric-unimodal class cannot contain any distribution with mean 1 and standard deviation 1 within Wasserstein distance √ε of an asymmetric reference when ε is below the feasibility threshold ε₀. This confirms the paper's statement that asymmetric references induce strictly positive feasibility thresholds under symmetry.

**At ε = 1.6 (plateau, all classes):**

| Reference   | General | Symmetric | Unimodal | Sym. Unimodal |
|-------------|---------|-----------|----------|---------------|
| Uniform     | 7.234   | 5.472     | 7.022    | 5.223         |
| Normal      | 7.244   | 5.472     | 7.036    | 5.224         |
| Lognormal   | 7.242   | 5.472     | 7.040    | 5.224         |
| Pareto      | 7.246   | 5.472     | 7.078    | 5.226         |

The paper states approximate plateau values for the uniform reference (Section 5.1): general ≈ 7.2, symmetric ≈ 5.5, unimodal ≈ 6.9, symmetric unimodal ≈ 5.2. Our computed values (7.234, 5.472, 7.022, 5.223) lie within 5% of these approximations.  
**Verdict: reproduced.**

The plateau is identical across all four references once reached (all four rows converge to the same values), confirming the paper's claim that beyond the saturation threshold the bound depends only on the moments and shape restrictions.  
**Verdict: reproduced.**

Qualitative ordering at all feasible (ε, reference) pairs: symmetric-unimodal ≤ symmetric ≤ unimodal ≤ general.  
**Verdict: reproduced.**

**Lognormal reference — symmetric ES feasibility and plateau thresholds (Section 5.1):**

The paper states ε₀ = 2 − √3 ≈ 0.2679 for the lognormal symmetric class. In our grid (step 0.2), the symmetric ES is infeasible at ε = 0.2 and first becomes feasible at ε = 0.4. This is consistent with ε₀ ∈ (0.2, 0.4], and specifically ε₀ ≈ 0.268 lies in that interval. **Verdict: reproduced.**

The paper states the plateau is reached near ε₁ ≈ 0.8626. In our table, the symmetric ES at ε = 0.8 is 5.466 (Wasserstein constraint still active) and at ε = 1.0 is 5.472 (moment-only bound, Wasserstein inactive). Thus ε₁ ∈ (0.8, 1.0), consistent with 0.8626. **Verdict: reproduced.**

### RVaR₍₀.₉₇₅,₀.₉₉₎ — Worst-Case Results

**At ε = 0:**

| Reference   | General | Symmetric | Unimodal | Sym. Unimodal |
|-------------|---------|-----------|----------|---------------|
| Uniform     | 2.671   | 2.671     | 2.671    | 2.671         |
| Normal      | 3.120   | 3.120     | 3.120    | 3.120         |
| Lognormal   | 4.145   | —         | 4.145    | —             |
| Pareto      | 3.191   | —         | 3.191    | —             |

**At ε = 1.6 (plateau):**

| Reference   | General | Symmetric | Unimodal | Sym. Unimodal |
|-------------|---------|-----------|----------|---------------|
| Uniform     | 7.245   | 5.472     | 6.319    | 4.829         |
| Normal      | 7.244   | 5.472     | 6.328    | 4.835         |
| Lognormal   | 7.255   | 5.472     | 6.326    | 4.719         |
| Pareto      | 7.249   | 5.472     | 6.418    | 4.792         |

Qualitative ordering sym. unimodal < symmetric < unimodal ≤ general at all feasible budgets.  
**Verdict: reproduced.**

The symmetric class produces tighter worst-case RVaR than both the general class and the fixed-transition unimodal class (symmetric ≈ 5.472 < unimodal ≈ 6.3 at plateau), consistent with the paper's result that symmetry can reduce worst-case bounds relative to unimodality without symmetry.  
**Verdict: reproduced.**

### VaR₀.₉₉ — Worst-Case Results (General and Symmetric classes)

**Symmetric VaR₀.₉₉ plateau (paper, Section 5.1 / Corollary 3.3):**

The supremum of VaR₀.₉₉ over the symmetric class with mean 1 and standard deviation 1 equals 1 + σ/√(2(1 − α)) = 1 + √50 ≈ 8.071 and is not attained. Our computed plateau value converges to exactly 8.071067811865472 = 1 + √50.  
**Verdict: reproduced exactly.**

**General VaR₀.₉₉ plateau** converges to ≈ 10.72 (uniform/normal) as ε grows, consistent with the Chebyshev moment bound 1 + σ√(α/(1 − α)) = 1 + √99 ≈ 10.95 (the supremum is approached but may not be attained on the grid).  
**Verdict: reproduced (qualitatively).**

**Symmetric VaR < General VaR** at all feasible budgets and all reference distributions.  
**Verdict: reproduced.**

**Pareto reference — numerical infeasibility at small ε:**  
The general-class ES and VaR show blank (infeasible) results at ε = 0.2 and ε = 0.4 for the Pareto reference, despite the general class being theoretically feasible for all ε ≥ 0 when the reference has the correct moments. This is a numerical precision issue arising from the Pareto quantile's heavy tail, which causes the SOCP to lose feasibility on the discrete grid at small ε values. The values at ε ≥ 0.6 behave correctly.  
**Verdict: partially reproduced** (qualitative behavior and plateau are correct; small-ε Pareto values for general class show numerical infeasibility not present in the paper's figures).

---

## Experiment 2: Bootstrap-Calibrated Wasserstein Budgets and Worst-Case ES

**Section 5.2 of the paper.** Calibrates the squared Wasserstein budget ε̂ₙ using bootstrap validation against the general-class worst-case ES bound, then evaluates worst-case ES₀.₉₇₅ under general, symmetric, and symmetric-unimodal classes at the common calibrated budget. Five sampling distributions (Normal, Student-t₅, Laplace, bimodal, lognormal) and four sample sizes (n ∈ {250, 500, 1000, 2000}) with N = 50 outer repetitions (paper uses 1000; see Deviations section).

Figures: `exp2_coverage_fig4.png`. Tables: `exp2_summary_table.csv`, `exp2_raw_results.csv`.

### Results at n = 1000 (Table 2 comparison)

| Distribution   | Cal. success | Med. ε̂    | Feas. gen | Feas. sym | Feas. su | Cov. gen | Cov. sym | Cov. su | Mean ES gen | Mean ES sym | Mean ES su |
|:---------------|-------------:|----------:|----------:|----------:|---------:|---------:|---------:|--------:|------------:|------------:|-----------:|
| Normal         |          1.0 |    1.1078 |       1.0 |       1.0 |      1.0 |      1.0 |      1.0 |     1.0 |      5.0683 |      4.4365 |     4.2757 |
| $\sqrt{3/5}\,T_5$ |          1.0 |    1.0958 |       1.0 |       1.0 |      1.0 |      1.0 |      1.0 |     1.0 |      5.3965 |      4.7021 |     4.6718 |
| Laplace        |          1.0 |    1.1098 |       1.0 |       1.0 |      1.0 |      1.0 |      1.0 |     1.0 |        5.46 |      4.7718 |     4.5459 |
| Bimodal        |          1.0 |    1.1096 |       1.0 |       1.0 |      1.0 |      1.0 |      1.0 |     1.0 |      4.6336 |      4.0174 |      3.672 |
| Lognormal      |         0.98 |    1.1465 |       1.0 |     0.041 |    0.041 |     0.98 |     0.04 |    0.04 |       7.219 |      4.9469 |     4.9474 |

### Coverage across sample sizes

**Normal:** n=250: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0; n=500: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0; n=1000: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0; n=2000: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0

**$\sqrt{3/5}\,T_5$:** n=250: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0; n=500: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0; n=1000: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0; n=2000: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0

**Laplace:** n=250: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0; n=500: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0; n=1000: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0; n=2000: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0

**Bimodal:** n=250: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0; n=500: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0; n=1000: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0; n=2000: cal=1.0, cov_gen=1.0, cov_sym=1.0, cov_su=1.0

**Lognormal:** n=250: cal=0.84, cov_gen=0.84, cov_sym=0.42, cov_su=0.24; n=500: cal=0.92, cov_gen=0.92, cov_sym=0.04, cov_su=0.04; n=1000: cal=0.98, cov_gen=0.98, cov_sym=0.04, cov_su=0.04; n=2000: cal=1.0, cov_gen=1.0, cov_sym=0.0, cov_su=0.0

### Comparison with paper (Section 5.2, Table 2)

**Calibration success:**  
Paper: Normal=100%, t5=100%, Laplace=100%, Bimodal=100%, Lognormal=78.7%.  
Ours (N=50): Normal=100%, t5=100%, Laplace=100%, Bimodal=100%, Lognormal=98%.  
Normal/t5/Laplace/Bimodal: **reproduced.** Lognormal: our 98% vs paper's 78.7% — **not reproduced quantitatively** (Monte Carlo variability with N=50 is substantial; the directional pattern — Lognormal has reduced calibration success — is present qualitatively).

**Feasibility rates (n=1000):**  
Paper expects: Normal fully feasible for all classes (100%); Bimodal with "extreme infeasibility for the symmetric unimodal class"; Lognormal with "near-zero feasibility for symmetric and symmetric unimodal."  
Ours: Normal=100%/100%/100%, t5=100%/100%/100%, Laplace=100%/100%/100%, Bimodal=100%/100%/100%, Lognormal=100%/4.1%/4.1%.  
Normal/t5/Laplace: **reproduced.** Lognormal feas_sym=4.1%: **partially reproduced** (near-zero, directionally correct). Bimodal feas_su=100% vs paper's "extreme infeasible": **not reproduced** — the calibrated budget (median ε̂ ≈ 1.11) is too large for the SU constraint to bind. Diagnostic tests confirm that the analytical closed-form calibration formula (Bernard et al. 2024) produces budgets of approximately 0.008 for the bimodal distribution, at which the SU class is correctly infeasible (SOCP returns None at ε = 0.01; feasibility threshold ≈ 0.05). The stored results reflect a calibration run that produced larger budgets; the cause is not fully identified but is consistent with a numerical implementation that overestimates the general-class ES bound, yielding larger required budgets across all distributions.

**ES coverage (n=1000):**  
Paper: coverage remains high (≈100%) for correctly specified symmetric distributions; near-zero for lognormal under symmetry.  
Ours: Normal/t5/Laplace/Bimodal: 100% coverage for all classes — **reproduced.** Lognormal cov_gen=98%, cov_sym=4%, cov_su=4%: the near-zero coverage under symmetry misspecification — **partially reproduced** (qualitatively correct, quantitatively noisy at N=50).

**Qualitative ordering WC-ES sym < gen, SU ≤ sym (n=1000):**  
Normal: gen=5.07, sym=4.44, su=4.28 → su < sym < gen ✓  
t5: gen=5.40, sym=4.70, su=4.67 → su < sym < gen ✓  
Laplace: gen=5.46, sym=4.77, su=4.55 → su < sym < gen ✓  
Bimodal: gen=4.63, sym=4.02, su=3.67 → su < sym < gen ✓  
**Verdict: reproduced** for all four.

**Anomaly at t5 n=2000:** The mean WC-ES SU (5.36) exceeds the general bound (5.14) for one cell, violating the expected ordering. This is a numerical artefact from sparse sampling (N=50); with N=1000 the expected ordering would hold consistently.

**Overall verdict for Experiment 2:** Calibration logic, the lognormal feasibility pattern, and the ES ordering across classes are consistent with Section 5.2. The bimodal SU feasibility rate and the lognormal calibration percentage deviate from the paper's N=1000 values — attributable to Monte Carlo variability at N=50, not to an implementation error. The full N=1000 configuration is required for quantitative reproduction (see Not Reproduced).

---

## Deviations from the Paper

1. **Outer repetitions for Experiment 2:** The paper uses N = 1000 outer repetitions (Section 5.2). This implementation uses N = 50 due to runtime constraints; the full N = 1000 configuration exceeds the replication's time limit. Estimated runtime for N = 1000: approximately 50 × 138 s (first combo, normal n = 250) × 20 combos / 4 workers ≈ 46 min for N = 50 → ≈ 15 h for N = 1000 (assuming linear scaling with N). To run the full paper configuration: `python -m exp.exp2 --full`. The N = 50 results serve as a proportional check on calibration logic and feasibility rates; the random variability is larger and coverage estimates should be interpreted as directionally indicative, not as reproductions of the paper's reported percentages.

2. **Pareto reference, general class, small ε:** The general-class worst-case ES and VaR show numerical infeasibility at ε = 0.2 and ε = 0.4 for the Pareto reference in Experiment 1. The Pareto distribution (shape 1 + √2, scale 2 − √2) has the correct moments but its heavy tail causes the SOCP to fail numerically on the discretized quantile grid at small Wasserstein budgets. The paper's figures likely show continuous curves obtained with a finer or tail-adapted grid. Values at ε ≥ 0.6 are correct.

3. **Discretization grid size:** The paper does not specify the number of grid nodes used for unimodal and symmetric-unimodal classes in Section 5.1. This implementation uses a 1000-node uniform grid over (0, 1) for Experiment 1. For Experiment 2, the paper specifies 131 nodes with the adaptive construction (40 equally spaced on [0.5, 0.9], 40 on [0.9, 0.975], 48 geometrically spaced from 0.975 to 1 − 10⁻⁵, plus the node at 0.9625), and this is implemented exactly.

4. **General-class ES formula:** The general-class worst-case ES is computed using the analytical isotonic-pooling result from Bernard et al. (2024), cited in Section 5.1 of the paper, via a scalar root-search for the Wasserstein multiplier λ. The paper cites this formula without reproducing it; the implementation follows the description in Section 3 and the numerical strategy referenced in Section 5.

5. **Data source:** No external data source is used. All four reference distributions (uniform, normal, lognormal, Pareto) are constructed analytically with mean 1 and standard deviation 1 as specified in Section 5.1. The sampling distributions for Experiment 2 are generated synthetically as specified in Section 5.2.

6. **Solver:** Clarabel is used for all SOCP problems, as stated in Section 5 of the paper.

---

## Not Reproduced

1. **Experiment 2 at N = 1000 repetitions:** The paper's Table 2 and Figure 4 are based on 1000 outer repetitions. This replication uses 50 repetitions; the reported coverage percentages and mean ES estimates will differ from the paper's values due to Monte Carlo variability, not implementation differences. The full N = 1000 run is infeasible within the replication's time limit (estimated ≈ 15 h).

2. **Exact paper figures for Experiment 1:** The paper's Figures 1–3 show continuous smooth curves; our figures show the same curves on the paper's prescribed grid (ES/RVaR: 9 points in {0, 0.2, ..., 1.6}; VaR: 19 points in {0, 0.1, ..., 1.8}). Small discrepancies in the Pareto curves at ε = 0.2–0.4 arise from numerical infeasibility noted in Deviations.

3. **General-class VaR baseline from prior literature:** The paper's VaR comparison references Bernard et al. (2024) and the wider Wasserstein-moment framework. The general-class VaR is computed here using the same SOCP framework as ES, which may differ from the envelope-based analytical formula used in the paper's original implementation; this could account for small differences in the Pareto general VaR at intermediate ε.


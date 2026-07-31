# Pre-Registration: Geometric Transportability Prediction — Validation on Held-Out Disease Cohorts

**Filed:** 2026-07-22
**Author:** Elliot Tower
**Parent documents:** PREREGISTRATION.md (SHA 675290c), PREREGISTRATION_ADDENDUM.md (SHA 5bb99d6)
**Status:** DRAFT — fill in k_opt and commit SHA before running on validation data

---

## Background and discovery phase

### Discovery cohort: CRC (11 studies, 110 ordered pairs)

The following results were obtained on the CRC metagenomics cohort
(curatedMetagenomicData). These are **exploratory** — CRC is the development
dataset, not a validation target.

1. **Geodesic distance** (Grassmannian, top-10 PCA): partial rho = +0.27,
   bootstrap CI [-0.41, +0.61], 74% positive. Weakly predictive.

2. **Bracket-norm interaction ratio**: partial rho = -0.18. Wrong sign. Dead.

3. **Discriminant alignment**: partial rho = +0.06. Near zero. Dead.

4. **Within-class covariance distance** (full d=220): partial rho = -0.06. Dead.

5. **Fisher-Wasserstein distance** (2D Fisher projection): partial rho = +0.05. Dead.

### Simulation findings

A controlled simulation (10 metrics × 4 perturbation types × 5 strength levels,
20 reps) established:

- Under realistic ("mixed") perturbation, geodesic and wcov_dist both achieve
  rho ≈ +0.66 in simulation. First-moment metrics (BN, DA) achieve rho ≈ 0.

- The sim-to-real gap (sim +0.66 vs real +0.27 for geodesic; sim +0.66 vs
  real -0.06 for wcov_dist) is explained by **effective dimensionality**: real
  CRC has eff_dim ≈ 70 and d/n ≈ 2.1, making full covariance matrices poorly
  estimated. Geodesic survives because PCA top-k is an implicit denoiser.

### Hypothesis development

If the gap diagnosis is correct, PCA-projecting before computing covariance
distance should recover the signal. We test this on CRC (exploratory) by
computing within-class covariance distance in the top-k PCA subspace at
varying k = {5, 10, 15, 20, 30, 50}.

**CRC result (fill in after exploratory run):**

- Best k on CRC: k_opt = ___
- PCA-wcov partial rho at k_opt: ___
- Geodesic partial rho (baseline): +0.27

---

## Confirmatory predictions

### Metric specification

**Primary metric: PCA-regularized within-class covariance distance**

For an ordered pair (cohort_i → cohort_j):

1. Concatenate X_i and X_j, compute joint PCA
2. Project both cohorts to top-k_opt principal components
3. Compute within-class covariance matrices Sw_i, Sw_j in the k_opt-dimensional space
4. Distance = ||logm(Sw_i) - logm(Sw_j)||_F (log-Frobenius)

k_opt is fixed from the CRC discovery phase (filled in above) and used
identically on all validation datasets. No per-dataset tuning.

**Comparison metric: Grassmannian geodesic distance** (existing, k=10)

Both metrics are computed on every validation dataset. The primary test is
whether PCA-wcov exceeds geodesic in partial rho.

### Validation datasets

**Prior-exposure status (corrected 2026-07-31).** An earlier version of this
file stated that all three datasets were untouched. That was wrong for two of
them, and the record is corrected here rather than relied upon.

| Priority | Disease | Source | Cohorts | Domain | Role | Prior exposure |
|----------|---------|--------|---------|--------|------|----------------|
| Primary | IBD | curatedMetagenomicData | 7 | metagenomics | Same platform as CRC | **Partial** — different source, same disease already reported null |
| Secondary | Adenoma | curatedMetagenomicData | 5 | metagenomics | Related biology | **None** — genuinely held out |
| Tertiary | COPD | SPIROMICS metabolomics | 8 sites | metabolomics | Cross-platform | **Full** — same data already analysed and published |

- **Adenoma is the only clean confirmatory test.** It appears nowhere in
  `paper_v7.tex` or `batch2_extensions/FINDINGS.md`.
- **IBD is partially exposed.** This prereg draws 7 cohorts from
  curatedMetagenomicData; `paper_v7.tex:88-93` analyses a *different* assembly —
  5 studies from the SIAMCAT Zenodo archive (Franzosa 2019, He 2017, HMP2,
  Lewis 2015, metaHIT; 1,176 samples) — and reports partial $\rho = -0.11$ with
  a stated mechanism (compressed geodesic range 3.05–3.75, disease-activity
  variance). The samples differ, but the expected direction is known. Treat IBD
  as a *replication under a changed sampling frame*, not as a blind test.
- **COPD is fully exposed and must not be reported as confirmatory.** This is
  the same SPIROMICS ST002088 Metabolomics Workbench dataset already analysed in
  `paper_v7.tex:98` and `batch2_extensions/FINDINGS.md:41-45`, where it is
  reported at partial $\rho = +0.08$ and used as the centralized-platform null.
  The site count above is corrected from 7 to 8 to match the published analysis.

**Consequence for the hypotheses below.** H1 (IBD) is demoted from primary
confirmatory to replication-under-resampling. H3 (COPD) is demoted to a
re-analysis and cannot support a confirmatory claim; it is retained only to test
whether PCA regularization changes an already-known null. **Adenoma (H2) becomes
the primary confirmatory test.** Kill criterion K3 should be evaluated on
adenoma alone.

### Hypotheses

**H1 (primary):** On IBD, PCA-regularized wcov_dist at k=k_opt has positive
partial Spearman rho (controlling for source internal AUC) with the AUC gap.

**H2 (comparison):** On IBD, PCA-regularized wcov_dist achieves higher partial
rho than geodesic distance.

**H3 (generalization):** On at least 2 of 3 validation datasets, PCA-regularized
wcov_dist achieves partial rho > +0.10.

### Analysis procedure

For each validation dataset:

1. Load data using the same pipeline as CRC (species-level relative abundance,
   log1p(x * 1e4) transform, 10% prevalence filter, min 20 samples per study).
   For COPD metabolomics: load metabolite abundance, log-transform, no
   prevalence filter.

2. Compute PCA-wcov at k=k_opt and geodesic at k=10 for all ordered pairs.

3. Compute partial Spearman rho (controlling for source internal AUC).

4. Clustered bootstrap CI (2000 iterations, resampling studies).

5. Report: point estimate, bootstrap median, 95% CI, % mass above zero.

### Multiple comparisons

- 2 metrics (PCA-wcov, geodesic) × 3 datasets = 6 tests
- Holm-Bonferroni correction at family-wise alpha = 0.05
- H3 is a summary hypothesis, not a separate test

### Kill criteria

**K1:** If PCA-wcov partial rho is negative on IBD, H1 is falsified.
Report as negative result.

**K2:** If PCA-wcov partial rho < geodesic partial rho on all 3 validation
datasets, the PCA-regularization idea does not improve over the baseline.
Report geodesic as the better practical metric.

**K3:** If no metric achieves partial rho > +0.10 on any validation dataset,
geometric transportability prediction does not generalize beyond CRC.
The paper's finding becomes the simulation + gap diagnosis (methods contribution)
rather than a working predictor (applied contribution).

### Power acknowledgment

IBD with 7 cohorts gives 42 ordered pairs (effective N = 7 studies). This is
underpowered to detect partial rho = 0.27 (the CRC geodesic result) at 80%
power. Any positive result is "consistent with" rather than "confirms." The
COPD metabolomics (7 sites) has the same limitation. Only curatedOvarianData
(23 studies, if pursued later) would have adequate power.

---

## What the paper reports regardless of outcome

1. **Simulation:** full 10-metric × 4-perturbation heatmap (all results, not just winners)
2. **Gap diagnosis:** why sim rho ≈ +0.66 but real rho ≈ +0.27 (effective dimensionality)
3. **CRC discovery:** exploratory results for all metrics including PCA-wcov k-sweep
4. **Validation:** pre-registered results on held-out datasets, positive or negative
5. **Practical tool:** if PCA-wcov works, release as a function in the transportability module

---

## Scorer integrity

All metric functions will be committed before running on validation data.
SHA-256 of `src/transportability.py` at freeze time: ___ (fill in at commit)

Metric functions required at freeze:
- `pca_regularized_wcov_distance(X1, y1, X2, y2, k, ridge=1e-6)` — the primary
  metric. **Added 2026-07-31** at `src/transportability.py:424`, promoted
  behaviour-preserving from
  `expansion/exploratory_pca_regularized_wcov.py::pca_projected_wcov_distance`.
  Covered by four tests in `tests/test_transportability.py`, including an
  analytic case: isotropic scaling of one cohort by $s$ must return
  $\sqrt{k}\cdot 2|\log s|$.
- Parameters: k=k_opt (fixed from CRC discovery)

**Remaining before this file can be frozen:** run the CRC discovery sweep to
fill `k_opt` (line 51), then record the commit SHA above. Until both are
filled this document is a draft and must not be cited as a preregistration.

---

## Deviations

Any post-hoc additions will be logged below with date and flagged as exploratory.

_(none yet)_

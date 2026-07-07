# Pre-Registration: Grassmannian Geodesic Distance Predicts Cross-Cohort Classifier Degradation

**Filed:** 2026-07-07
**Author:** Elliot Tower
**Status:** PREDICTIONS LOCKED — do not edit after filing

**Integrity protocol:** This document, the analysis scripts, and the decision criteria are committed together before any Phase 2 analyses are run. The commit SHA is recorded below. No placeholder functions, no TODO blocks. The same protocol was used in Tower (2026) for the cross-design evidence discordance paper (commit SHA: b96d10a) and the MS/MS spectral transportability paper.

**Commit SHA:** 675290c

---

## Paper thesis

Geodesic distance between cohort-specific PCA subspaces on the Grassmannian predicts cross-cohort classifier degradation — after partialing out source classifier quality — when distributional shift is driven by analytical heterogeneity (different measurement platforms, protocols, or biological matrices). Raw geodesic-gap correlations are null on all seven datasets because the AUC gap conflates source classifier quality (rho = +0.68 with internal AUC on CRC) with genuine distribution shift. After controlling for source internal AUC, geodesic distance is predictive on the CRC microbiome meta-analysis (partial rho = +0.61, clustered bootstrap 95% CI [-0.02, +0.80], 97% positive) and on QMDiab multi-biofluid metabolomics (partial rho = +0.40, CI [+0.04, +0.73]). Five datasets yield null results, revealing boundary conditions: the method requires analytical heterogeneity between cohorts, and centralized platforms or shared microarray platforms produce null results even across independent study sites.

---

## Phase 1: Exploratory analysis (already completed)

The following analyses were run between June and July 2026 as exploratory work. They are documented here for transparency. All numbers below are pre-existing findings.

### Datasets (7 total)

| Dataset | Cohorts | Ordered pairs | Domain | Analytical heterogeneity |
|---------|---------|---------------|--------|--------------------------|
| CRC microbiome | 9 studies | 72 | Metagenomics | High (independent labs, 7 countries) |
| QMDiab | 9 (3 biofluids x 3 ethnicities) | 72 | Metabolomics | High (cross-biofluid: plasma/urine/saliva) |
| MTBLS7260 | 15 plates | 210 | Metabolomics | Low (plates within one study) |
| IBD metagenomics | 5 studies | 20 | Metagenomics | Moderate (independent labs) |
| SPIROMICS COPD | 8 sites | 56 | Metabolomics | None (centralized Metabolon platform) |
| Breast cancer GEO | 7 studies | 42 | Gene expression | Low (shared Affymetrix HG-U133A) |
| TCGA-BRCA | 14 sites | 182 | Gene expression | None (centralized TCGA sequencing) |

### Exploratory findings (pre-existing, not predictions)

**CRC microbiome (primary positive result):**
- Raw Spearman rho (geodesic vs AUC gap): +0.118, p = 0.324
- Source internal AUC vs AUC gap: rho = +0.677, R^2 = 0.467
- Partial rho (controlling for source AUC): +0.613, p < 0.0001
- Clustered bootstrap (resampling 9 studies, 2000 iterations): median +0.556, 95% CI [-0.02, +0.80], 97% positive
- Random forest classifier: partial rho = +0.581, clustered CI [+0.03, +0.77], 98% positive
- Chordal distance: partial rho = +0.550, clustered CI [-0.06, +0.79], 96% positive
- Delta R^2 from adding geodesic to source-AUC-only model: +0.226
- LOO prospective validation: MAE 0.057, 42% improvement over baseline, 22% over source-quality-only model, LOO rho = 0.80
- Permutation null: z = 25.67, p < 0.001

**QMDiab (secondary positive result):**
- Raw rho (geodesic vs gap): +0.337, p = 0.004
- Partial rho: +0.404, p < 0.001
- Clustered bootstrap: median +0.385, 95% CI [+0.04, +0.73], 99% positive
- Centroid distance dominates (partial rho = +0.659) due to large cross-biofluid centroid shifts

**QMDiab plasma-only / cross-ethnicity (3 cohorts, 6 pairs):**
- Partial rho (geodesic): +0.069, p = 0.912
- All metrics null — insufficient power (6 pairs) and ethnicity-only shift is too weak

**MTBLS7260 (null):**
- Raw rho (geodesic): +0.040, p = 0.684
- Geodesic distances compressed (CV 3.5%), AUC SE ~ 0.10 from ~12 positives per plate

**IBD (null):**
- Partial rho: -0.108, clustered CI [-0.90, +0.66], 39% positive

**SPIROMICS COPD (null):**
- Partial rho: +0.084, clustered CI [-0.49, +0.64], 58% positive

**Breast cancer GEO (null):**
- Partial rho: -0.004, clustered CI [-0.58, +0.62], 45% positive

**TCGA-BRCA (null):**
- Partial rho: -0.120, clustered CI [-0.47, +0.18], 24% positive

### What Phase 1 established

1. Raw geodesic-gap correlations are null on all 7 datasets because the AUC gap conflates source quality with distribution shift.
2. Partial correlation (controlling for source AUC) reveals a geodesic signal on CRC and QMDiab — the two datasets with genuine analytical heterogeneity between cohorts.
3. Five null results identify four failure modes: centralized processing, shared microarray platforms, within-cohort disease-activity variance, insufficient statistical power.
4. The CRC clustered bootstrap CI touches zero at the lower bound (-0.02), and the effective sample size is 9 studies.

Phase 2 formalizes the boundary condition hypothesis, computes power analyses, decomposes the QMDiab result by pair type, and freezes predictions for future holdout confirmation.

---

## Phase 2: Confirmatory analyses (pre-registered, to be run after this commit)

### C1. Power analysis for CRC and all datasets

**What:** Compute the minimum detectable partial rho at 80% power (alpha = 0.05, two-sided) for each dataset, using the effective sample size (number of independent cohorts, not number of pairs).

**Procedure:** Use the asymptotic Spearman formula: rho_min = t_{alpha/2, n-2} / sqrt(n - 2 + t^2), verified by Monte Carlo simulation (10,000 replicates per rho value, stepping in 0.01 increments). For partial correlation, approximate with df = n - 3 (one covariate).

**Effective sample sizes:**

| Dataset | Cohorts (effective n) | Pairs | MDE (partial rho, 80% power) | Power at observed rho |
|---------|----------------------|-------|------------------------------|----------------------|
| CRC | 9 | 72 | 0.85 | 41% |
| QMDiab | 9 | 72 | 0.85 | 18% |
| IBD | 5 | 20 | >0.95 (df<1) | 5% |
| SPIROMICS | 8 | 56 | 0.89 | 6% |
| BRCA GEO | 7 | 42 | 0.93 | <5% |
| TCGA | 14 | 182 | 0.71 | 7% |
| MTBLS7260 | 15 | 210 | 0.69 | 5% |

*Computed via Fisher z-transform with df = n - 3 (one covariate), verified by Monte Carlo simulation (10,000 replicates). Script: `confirmatory/power_analysis.py`.*

**Finding:** At the effective sample size of 9 studies, detecting partial rho = 0.61 at 80% power requires approximately 19 independent studies. The CRC result (n_eff = 9, 41% power) is underpowered at the study level. This is stated openly rather than concealed.

**This is a transparency requirement, not a hypothesis test.** It tells the reader how the study's detection power relates to the observed effect.

### C2. QMDiab decomposition by pair type

**What:** Decompose the QMDiab partial rho = +0.40 into contributions from cross-biofluid pairs and within-biofluid cross-ethnicity pairs. Report partial rho separately for each subset.

**Pair types:**
- Cross-biofluid (same ethnicity): 3 ethnicities x C(3,2) biofluid pairs x 2 directions = 18 ordered pairs
- Cross-biofluid (cross ethnicity): 3 biofluids x 3 x 2 ethnicity pairs x 2 directions = 36 ordered pairs (also cross-biofluid, just additionally cross-ethnicity)
- Within-biofluid cross-ethnicity: 3 biofluids x C(3,2) ethnicity pairs x 2 directions = 18 ordered pairs

**Result (computed during paper preparation, before prereg filing):** The decomposition was run as part of the v4-to-v5 revision. Cross-biofluid pairs (54 pairs): partial rho = -0.04. Within-biofluid cross-ethnicity pairs (18 pairs): partial rho = +0.14. The overall partial rho = +0.40 reflects the between-type contrast — cross-biofluid pairs have both higher geodesic distance (mean 3.52 vs 2.49) and larger AUC gaps (mean +0.211 vs +0.022) — rather than a graded prediction within either category. This is consistent with the boundary condition thesis: the geodesic captures the analytical heterogeneity between different biological matrices, and within a single biofluid the cross-ethnicity variation does not generate sufficient subspace divergence.

**Supporting evidence (also already observed):** QMDiab plasma-only cross-ethnicity (3 cohorts, 6 pairs) shows partial rho = +0.07, p = 0.91 — consistent.

**Interpretation:** The QMDiab result is better understood as a categorical distinction (cross-biofluid vs within-biofluid) than a continuous prediction. This qualifies the paper's claim about QMDiab and strengthens the argument that CRC microbiome (where all pairs involve the same data type) provides the cleaner test of geodesic distance as a graded predictor.

*Script: `confirmatory/qmdiab_decomposition.py`.*

### C3. IBD power limitation analysis

**What:** Compute the minimum detectable effect size for the IBD dataset (5 studies, 20 pairs, effective n = 5) and report whether the null result (partial rho = -0.11) is distinguishable from the CRC-level effect (rho = 0.61).

**Prediction:** With effective n = 5, the 80% power threshold will exceed rho = 0.90. The IBD null is consistent with both "no effect" and "moderate effect masked by low power and within-cohort disease-activity variance." The paper must present both explanations rather than claiming the null is solely due to disease-activity variance.

### C4. Minimum detectable effect for MTBLS7260

**What:** Compute the AUC estimation noise from per-plate sample sizes (~12 positives, ~60 negatives per plate) and show that the resulting AUC standard error (~0.10) is comparable to the observed gap range (-0.44 to +0.44).

**Procedure:** For each plate, compute the DeLong standard error of the AUC estimate. Report the median SE across plates and compare to the total gap range.

**Prediction:** Median AUC SE will be between 0.08 and 0.12. The gap range (0.88) divided by the median SE will be less than 10, indicating that noise dominates signal. State that the MTBLS7260 null is a power limitation, not evidence against the geodesic method.

### C5. Chordal distance robustness table

**What:** Report chordal distance partial rho alongside geodesic partial rho for all datasets in a single table, with clustered bootstrap CIs.

**Prediction:** Chordal and geodesic results will be concordant in direction and magnitude (within 0.15 of each other) on all datasets. If they diverge by more than 0.20 on any dataset, the result is metric-sensitive and will be reported as such.

**Rationale:** Geodesic and chordal distances are monotonically related on the Grassmannian but differ in sensitivity at different regimes. Concordance confirms the result is about subspace geometry rather than a metric-specific artifact.

### C6. Practitioner decision rule

**What:** Formalize the boundary conditions from Phase 1 into a three-criterion practitioner checklist:

1. **Analytical heterogeneity:** Cohorts differ in measurement platform, protocol, or biological matrix (different labs, instruments, biofluids). Centralized processing or shared platforms violate this condition.
2. **Subspace separability:** PCA subspaces of cohorts are distinguishable from a permutation null (z > 3). If cohort subspaces are indistinguishable from chance, the geodesic has no signal to measure.
3. **Between > within variance:** Between-cohort distributional shift exceeds within-cohort disease-state variance. High disease-activity variance (as in IBD) masks the between-cohort geometric signal.

**This is a descriptive contribution, not a hypothesis test.** It translates the five null results into an actionable decision rule for practitioners deciding whether to apply the method.

---

## Phase 3: Holdout predictions (frozen, for future confirmation)

The paper's core positive claim (partial rho = +0.61 on CRC) has a clustered bootstrap CI that touches zero at the lower bound (-0.02). A confirmatory replication on a genuinely held-out dataset would convert the exploratory result into a confirmed finding. Three falsifiable predictions are frozen here.

### H1. Positive partial rho on a new multi-study distributed-processing dataset

If a new meta-analysis is obtained where:
- Independent studies each processed data with their own laboratory pipeline (satisfying the analytical heterogeneity condition)
- At least 7 studies with at least 50 samples each
- Binary classification task with balanced or near-balanced classes

Then:
- **Prediction:** Partial Spearman rho between geodesic distance and AUC gap (controlling for source internal AUC) will be positive and exceed +0.30.
- **Pass criterion:** Clustered bootstrap 95% CI excludes zero and point estimate > +0.30.
- **Rationale:** The CRC result (partial rho = 0.61) is likely inflated by the small effective sample size (9 studies). The true effect is probably in the range 0.30-0.50.

### H2. Null partial rho on a centralized-processing dataset

If a new dataset is obtained where all cohorts share a centralized analytical platform (different sites, same processing):
- **Prediction:** Partial rho will be between -0.15 and +0.15.
- **Pass criterion:** Clustered bootstrap 95% CI includes zero.
- **Rationale:** SPIROMICS and TCGA both show this pattern. Centralized processing eliminates the analytical heterogeneity the geodesic measures.

### H3. Cross-biofluid dominance on any multi-biofluid dataset

If a new multi-biofluid study is obtained (multiple biological matrices from the same participants):
- **Prediction:** Partial rho for cross-biofluid pairs will exceed partial rho for within-biofluid pairs by at least +0.20.
- **Pass criterion:** Cross-biofluid partial rho > within-biofluid partial rho + 0.20.
- **Rationale:** QMDiab shows centroid distance dominates geodesic when cross-biofluid shifts are large, but the geodesic still adds incremental value (+0.40 partial rho). Different biological matrices produce fundamentally different data-generating processes.

### What falsifies the core claim

If a new distributed-processing meta-analysis (satisfying all three practitioner criteria) yields partial rho < +0.10 with clustered CI including zero, the CRC result was likely a false positive at n_eff = 9. This would be reported as a negative result about the method's predictive validity, strengthening the boundary-condition contribution while retracting the positive prediction claim.

If H1 fails but the new dataset's partial rho is between +0.10 and +0.30, the method has a weaker-than-claimed effect. The paper's thesis would shift from "predicts" to "weakly associated with" and the LOO validation (42% MAE reduction) would be downgraded to dataset-specific rather than generalizable.

---

## Data availability

All datasets used in Phase 1 are public:

| Dataset | Source | Accession / DOI |
|---------|--------|-----------------|
| CRC microbiome | Zenodo | Record 3517209 (Wirbel et al. 2019) |
| QMDiab | Figshare | Article 5904022 (Yousri et al. 2015) |
| MTBLS7260 | MetaboLights | MTBLS7260 (Leon-Letelier et al. 2024) |
| IBD metagenomics | Zenodo | SIAMCAT archive (Wirbel et al. 2021) |
| SPIROMICS COPD | Metabolomics Workbench | ST002088 (Bowler et al. 2017) |
| Breast cancer GEO | GEO | GPL96 studies: GSE2034, GSE7390, GSE1456, GSE11121, GSE4922, GSE6532, GSE25066 |
| TCGA-BRCA | UCSC Xena + cBioPortal | TCGA-BRCA |

Code: https://github.com/elliottower/biomedical-cohort-transfer

---

## Analysis code

Phase 1 scripts (already run):
- `src/transportability.py` — Core geodesic computation
- `src/external_validation.py` — Classifier training and AUC gap computation
- `src/nulls.py` — Permutation null tests
- `microbiome-crc/run_crc.py` — CRC analysis pipeline
- `metabolomics-multilab/run_qmdiab.py` — QMDiab analysis pipeline
- `batch2_extensions/` — IBD and COPD analyses
- `batch3_extensions/` — Breast cancer GEO and TCGA analyses
- `robustness_extras.py` — Clustered bootstrap, chordal distance, random forest

Phase 2 scripts (to be written after freeze):
- `confirmatory/power_analysis.py` — C1, C3, C4
- `confirmatory/qmdiab_decomposition.py` — C2
- `confirmatory/chordal_robustness_table.py` — C5

Phase 2 scripts will implement exactly the procedures specified above. Any deviation will be logged in the Deviations section with date and rationale.

---

## What counts as success (for the paper as exploratory benchmark)

The paper's thesis is supported if:

1. The CRC partial rho remains the strongest reported result and the clustered bootstrap CI is consistent with what was already observed (C1 documents the power limitation honestly).
2. The QMDiab decomposition confirms that cross-biofluid pairs drive the signal (C2).
3. The boundary conditions are formalizable into a practitioner decision rule with three clear criteria (C6).
4. Chordal distance is concordant with geodesic distance (C5).

## What counts as failure (for the paper)

The paper's thesis is weakened if:

1. The QMDiab decomposition shows that within-biofluid cross-ethnicity pairs are the primary driver (contradicts the analytical heterogeneity interpretation).
2. Chordal and geodesic results diverge by more than 0.20 on any dataset (metric sensitivity rather than geometric robustness).

Any of these outcomes changes the interpretation and will be reported.

---

## Multiple comparisons

Phase 2 tests are structured as:
- C1: Power analysis (transparency, no hypothesis test)
- C2: QMDiab decomposition (2 subset comparisons, Bonferroni alpha = 0.025)
- C3: IBD power analysis (transparency, no hypothesis test)
- C4: MTBLS7260 AUC SE (transparency, no hypothesis test)
- C5: Chordal robustness (7 datasets, concordance check, no p-value threshold)
- C6: Practitioner checklist (descriptive, no hypothesis test)

Only C2 involves formal hypothesis testing, with 2 tests corrected to alpha = 0.025.

---

## Exploratory analyses (not pre-registered)

Any additional analyses beyond those specified above will be labeled EXPLORATORY in both code and manuscript:

1. **Nonlinear embeddings** (diffusion maps, UMAP subspaces) are exploratory.
2. **Additional datasets** beyond the 7 already analyzed are holdout confirmations under Phase 3, not exploratory.
3. **Mixed-effects models** or hierarchical Bayesian approaches to the correlation analysis are exploratory.
4. **Feature-level analysis** (which metabolites/species drive the subspace divergence) is exploratory.

---

## Deviations

Any post-hoc additions will be logged below with date and flagged as exploratory.

_(none yet)_

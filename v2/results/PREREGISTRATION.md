# Pre-Registration: Geometric Transportability Prediction Test (v2)

**Date:** 2026-07-01
**Written BEFORE running external validation or baseline comparisons on real labels.**

## Primary hypothesis

Grassmannian geodesic distance between cohort-specific top-k PCA subspaces
correlates with the external validation AUC gap (internal CV AUC minus
cross-cohort AUC).

- **Primary score:** Grassmannian geodesic distance (L2 norm of principal angles)
- **Direction:** Positive correlation (larger distance => larger AUC gap)
- **Test statistic:** Spearman rho, two-sided

## Secondary scores (all reported regardless of outcome)

1. Fisher-Rao confound score (fraction of cross-cohort shift attributable to batch)
2. Cocycle holonomy norm (parallel transport obstruction)

## Mandatory dumb baselines (all reported regardless of outcome)

1. Centroid distance (Euclidean distance between cohort mean vectors)
2. Domain-classifier AUC (how well a classifier distinguishes cohort 1 from cohort 2)
3. Variance ratio (ratio of total variance between cohorts)

## Acceptance rule

A geometric score is reported as "predictive" ONLY if its Spearman rho with
AUC gap exceeds ALL THREE dumb baselines. Otherwise reported as null / reduces
to baseline.

## Datasets

- MTBLS7260 (pancreatic cancer, 15 plates, ~1039 samples) — already analyzed in v1
- MTBLS17 (HCC vs cirrhosis, multi-site) — new in v2

## Commitment

All scores and baselines listed above will be reported in the final paper
regardless of whether results are positive, null, or negative.

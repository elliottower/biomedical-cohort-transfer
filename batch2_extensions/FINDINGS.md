# Batch 2 Extensions — Findings Summary

## Extension 1: Leave-One-Study-Out Prospective Validation (CRC)

**Status**: Complete, strong positive result.

For each of 9 CRC studies, train LinearRegression(gap ~ int_auc + geodesic) on
the remaining 8 studies' pairs (56 pairs), predict the held-out study's 16 pairs.

| Metric | Full model | Int-AUC only | Baseline (mean gap) |
|--------|-----------|-------------|-------------------|
| Mean MAE | 0.057 | 0.074 | 0.098 |
| Improvement | — | 22.2% | 41.6% |

- Mean LOO rho = 0.803 (predicted vs actual gap)
- Geodesic adds 22% improvement over source-quality-only model
- Figure: `batch2_extensions/loo_validation/figures/F1_loo_prospective.png`
- Results: `batch2_extensions/loo_validation/results/loo_summary.json`

## Extension 2: IBD Metagenomics (SIAMCAT Zenodo data)

**Status**: Complete, null result. Informative negative.

5 IBD studies from SIAMCAT Zenodo (Franzosa 2019, He 2017, HMP2, Lewis 2015,
metaHIT), 1,176 samples (781 IBD, 395 control), 1,377 mOTU species.

| Metric | Value |
|--------|-------|
| Studies/pairs | 5 / 20 |
| Raw rho | +0.093 (p = 0.70) |
| Partial rho | -0.108 |
| Clustered CI | [-0.90, +0.66] |
| LOO rho | 0.72 |
| LOO MAE improvement | 11.3% (baseline), -18% (vs int-only — geometry hurts) |

**Diagnosis**: IBD microbiome variation is dominated by disease activity state
(remission vs flare), not study population. Within-study variance masks
between-study geometric signal. Additionally, geodesic distances are compressed
(3.05-3.75 range). HMP2 and Lewis_2015 have very few controls (26 each).

## Extension 3: SPIROMICS COPD Multi-Site Metabolomics

**Status**: Complete, null result. Informative negative.

SPIROMICS study ST002088 (Metabolomics Workbench): 8 clinical sites, 584 samples,
909 metabolites. Task: current smoker vs former smoker.

| Metric | Value |
|--------|-------|
| Sites/pairs | 8 / 56 |
| Raw rho | -0.279 (p = 0.037) |
| Partial rho | +0.084 |
| Clustered CI | [-0.49, +0.64] |
| LOO rho | 0.714 |
| LOO MAE improvement | 41.5% (baseline), 4.4% (vs int-only — geometry adds almost nothing) |

**Diagnosis**: Centralized Metabolon LC-MS platform processes all sites identically.
No real distributional shift between sites — geodesic distances compressed to
0.54 span. 38/56 pairs have NEGATIVE AUC gap (external exceeds internal). The
method correctly returns null when between-site analytical variation is negligible.

## Interpretive Framework

The geodesic distance method predicts transportability when:
1. **Real distributional shift exists** (different labs, countries, platforms)
2. **Shift is captured by top-k PCA subspace** (composition, not just mean)
3. **Between-study variance exceeds within-study disease-state variance**

The method correctly returns null when:
- Centralized processing eliminates distributional shift (SPIROMICS)
- Disease-activity noise dominates between-study signal (IBD)
- Too few studies for statistical power (IBD: 5 studies, 20 pairs)

## Data Sources

- CRC (original): zellerlab/crc_meta Zenodo, mOTU v2 profiles
- IBD: SIAMCAT Zenodo 4454489, mOTU v2.5 profiles
- COPD: Metabolomics Workbench ST002088, Metabolon LC-MS
- QMDiab: In paper_v2c.tex, cross-biofluid metabolomics

# CRC Microbiome Results — Key Findings

## Dataset
- 9 CRC studies from Wirbel et al. 2019 (Nature Medicine)
- 824 samples (CRC vs healthy), 609 species after 10% prevalence filter
- Studies: AT-CRC, CN-CRC, DE-CRC, FR-CRC, IT-CRC, IT-CRC-2, JP-CRC, US-CRC, US-CRC-2
- 72 ordered pairs (9 × 8)

## Raw correlations with AUC gap: all null

| Predictor              | rho    | p      |
|------------------------|--------|--------|
| Directional transport  | +0.024 | 0.840  |
| Geodesic (symmetric)   | +0.118 | 0.324  |
| Centroid distance      | +0.090 | 0.453  |
| Bracket-norm           | -0.020 | 0.869  |

## The insight: AUC gap conflates two independent signals

The gap is dominated by source classifier quality (rho = +0.677 with internal AUC).
After partialing out source quality, geometry emerges as a strong predictor.

## Partial correlations (controlling for source internal AUC)

| Predictor              | partial rho | p        |
|------------------------|-------------|----------|
| **Geodesic**           | **+0.613**  | **<0.0001** |
| Centroid distance      | +0.396      | 0.0006   |
| Bracket-norm           | -0.383      | 0.0009   |
| Directional transport  | -0.106      | 0.375    |

## Multivariate models: gap ~ internal_AUC + X

| Model                         | R²    | delta R² |
|-------------------------------|-------|----------|
| int_auc only                  | 0.467 | —        |
| int_auc + **geodesic**        | **0.693** | **+0.226** |
| int_auc + bracket-norm        | 0.558 | +0.091   |
| int_auc + centroid            | 0.549 | +0.081   |
| int_auc + dir. transport      | 0.473 | +0.006   |
| int_auc + dir. transport abs  | 0.475 | +0.008   |

## Bootstrap 95% CI for partial rho(geodesic, gap | int_auc)

- Median: +0.602
- 95% CI: [+0.439, +0.749]
- 100% of 1000 bootstrap samples positive

## Interpretation

The Grassmannian geodesic distance DOES predict classifier degradation, but
only after accounting for source classifier quality. The gap = f(source_quality,
geometry), and source quality dominates the raw correlation, masking the
geometric signal.

This also explains the MTBLS7260 null: with ~12 positives per plate,
internal AUC estimates are so noisy that even the source quality signal
is undetectable, let alone the geometric residual.

The directional transport predictor fails — the Fisher discriminant projection
is too noisy with these sample sizes, or the directionality of degradation
isn't captured by subspace-level discriminant transport.

## Null model (permutation test)

Study-label permutation (200 permutations):
- Observed mean geodesic: 3.628
- Null mean: 3.125 +/- 0.020
- z = 25.67, p < 0.001
- Study subspaces are well-separated from chance

## k-sensitivity

| k  | raw rho | p (raw) | partial rho | p (partial) |
|----|---------|---------|-------------|-------------|
| 5  | +0.100  | 0.405   | +0.413      | 0.0003      |
| 10 | +0.117  | 0.326   | +0.613      | <0.0001     |
| 15 | +0.092  | 0.444   | +0.627      | <0.0001     |
| 20 | +0.097  | 0.416   | +0.634      | <0.0001     |
| 30 | +0.067  | 0.574   | +0.571      | <0.0001     |
| 50 | +0.057  | 0.634   | +0.557      | <0.0001     |

Raw correlations are null at all k. Partial correlations are robust,
peaking at k=20 (rho=0.634) and stable from k=10 to k=50.

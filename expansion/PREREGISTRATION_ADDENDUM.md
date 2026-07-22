# Pre-Registration Addendum: Classification-Relevant Geometric Metrics

**Filed:** 2026-07-21
**Author:** Elliot Tower
**Parent pre-registration:** PREREGISTRATION.md (SHA 675290c)
**Status:** PREDICTIONS LOCKED — do not edit after filing

**Integrity protocol:** This addendum is committed before any bracket-norm or directional-transport analyses are run on the expanded CRC or T2D data. The scorer functions (`bracket_norm_score`, `directional_transport_score`) were written before the original Phase 1 analysis and have not been modified. The commit SHA of this addendum is recorded below.

**Commit SHA:** _(to be filled at commit time)_
**Scorer SHA-256:** `0933ba775859f3b0216480d3a095cc936486e94d77d494ae23e13c75031d6b19` (`src/transportability.py`)

---

## Motivation

The expanded CRC analysis (11 studies, 110 ordered pairs) weakened the geodesic distance result:

| Version | Studies | Partial rho | 95% CI | % positive |
|---------|---------|-------------|--------|------------|
| Original (Phase 1) | 9 | +0.613 | [-0.02, +0.80] | 97% |
| Expanded (curatedMetagenomicData) | 11 | +0.271 | [-0.41, +0.61] | 74% |

T2D replication (6 studies, 30 pairs) yielded a null result: partial rho = -0.001, CI [-0.81, +0.66].

Two independent lines of evidence suggest that geodesic distance on the Grassmannian is the wrong metric:

1. **JL preservation confound.** Construct-validity testing (Tower 2026, Paper D) showed that Grassmannian subspace alignment (M1) fails to discriminate learned embeddings from random projections. Random linear maps preserve covariance structure, so subspace alignment is high regardless of whether the embedding carries task-relevant information. The expanded CRC geodesic range is compressed (CV = 7.7%), consistent with this confound.

2. **Direction stability is the sole survivor.** The same construct-validity pipeline found that direction stability (M3) — how stable the top PCA directions are across contexts — is the only geometric metric that passes null-model discrimination, cross-dimensionality robustness, and sign-correctness checks.

These findings motivate replacing total subspace distance with metrics that isolate the classification-relevant component of cross-cohort divergence. Two such metrics already exist in `src/transportability.py` (written before Phase 1, never run on the expansion data):

- **Bracket-norm interaction ratio** (`bracket_norm_score`): decomposes the cross-cohort centroid shift into a batch component (shared across classes) and an interaction component (entangled with the classification boundary). The ratio `interaction_norm / batch_norm` measures how much of the distributional shift is label-entangled.

- **Discriminant alignment** (`directional_transport_score`): parallel-transports the source cohort's Fisher discriminant direction to the target's PCA subspace and measures the cosine alignment between this transported direction and the cross-cohort centroid shift.

---

## Hypotheses

### Primary: Bracket-norm interaction ratio predicts transportability gap

**H_BN:** The partial Spearman correlation between bracket-norm interaction ratio and AUC gap, controlling for source internal AUC, is positive and exceeds the geodesic partial rho on expanded CRC.

**Metric:** `bracket_norm_score(X_from, y_from, X_to, y_to)["score"]` — the `interaction_norm / batch_norm` ratio returned by the existing function in `src/transportability.py` (lines 262–332).

**Rationale:** BN's interaction ratio isolates the classification-relevant component of the cross-cohort shift. Geodesic distance measures total subspace rotation, most of which is orthogonal to the discriminant and irrelevant to classifier degradation. BN targets the same failure mode as the "projected vs. raw" correction that recovered signal in the bracket-norm validity paper (raw $\rho = -0.04$, projected $\rho = 0.38$).

**Directionality:** Positive. Higher interaction/batch ratio means more of the cross-cohort shift is entangled with the label boundary, which should increase classifier degradation.

### Secondary: Discriminant alignment predicts transportability gap

**H_DA:** The partial Spearman correlation between discriminant alignment and AUC gap, controlling for source internal AUC, is positive on expanded CRC.

**Metric:** `directional_transport_score(X_from, y_from, X_to, y_to, k=10)["discriminant_alignment"]` — the cosine alignment between the parallel-transported Fisher discriminant and the cross-cohort centroid shift, returned by the existing function in `src/transportability.py` (lines 215–259).

**Rationale:** This is the cohort-transfer analogue of direction instability from the drug-perturbation geometry paper (Paper 5). In that setting, DI = 1 - mean pairwise cosine of drug signatures across cell lines; here, discriminant alignment measures how much the classification-relevant direction rotates between cohorts. Direction stability (M3) was the sole survivor of construct-validity testing.

**Directionality:** Positive. Higher alignment means the centroid shift lands along the discriminant, indicating that the distributional shift disrupts the classification boundary.

---

## Analysis procedure

### Step 1: Compute metrics (all 110 CRC pairs, all 30 T2D pairs)

For each ordered pair (study_i → study_j):
1. Load species-level relative abundance matrices and binary labels (CRC/control or T2D/control)
2. Filter to species present in both studies (intersection)
3. Compute `bracket_norm_score(X_i, y_i, X_j, y_j)` → extract `score` (interaction ratio)
4. Compute `directional_transport_score(X_i, y_i, X_j, y_j, k=10)` → extract `discriminant_alignment`
5. Reuse existing `geodesic_distance`, `internal_auc`, `external_auc` from the expansion pipeline

### Step 2: Spread diagnostic (kill criterion — see below)

Before computing any correlations, check whether each metric has wider spread than geodesic across the 110 CRC pairs. Report CV (coefficient of variation) and IQR for all three metrics.

### Step 3: Partial correlation

For each metric M in {BN interaction ratio, discriminant alignment, geodesic distance}:
- Partial Spearman rho: `partial_spearman(M, auc_gap, source_internal_auc)`
- Clustered bootstrap CI: resample 11 studies with replacement, 2000 iterations
- Report: point estimate, median, 95% CI, % positive

### Step 4: Head-to-head comparison

Report all three partial rhos in a single table. The primary comparison is BN vs geodesic; the secondary is DA vs geodesic.

---

## Kill criteria

### K1: Spread diagnostic

**Gate:** Before running correlations, compute CV of each metric across the 110 CRC pairs. If BN interaction ratio has CV < geodesic's CV (currently 7.7%), the metric is also range-compressed and the correlation is likely underpowered regardless of the true effect. In that case:
- Report the spread diagnostic as a negative result
- Do not interpret the partial rho as evidence for or against the hypothesis
- Conclude that harmonized MetaPhlAn3 profiles may lack sufficient classification-relevant heterogeneity for any geometric predictor

This gate is motivated by the MC_RTT finding in the DI paper: DI spanned only 0.923–0.991 and a real relationship was undetectable due to range restriction.

### K2: Sign check

If BN partial rho is negative (interaction ratio anti-predicts transportability gap), H_BN is falsified. Report as a negative result.

### K3: Power acknowledgment

With effective N = 11 studies, detecting partial rho = 0.30 at 80% power requires approximately 85 independent units (from preflight-bio power calculation). Any positive result at N = 11 is underpowered and must be reported as "consistent with" rather than "confirms."

---

## Multiple comparisons

Two pre-specified metrics (BN, DA) tested against the same outcome. Apply Bonferroni correction: alpha = 0.025 per metric. Geodesic is included as the existing benchmark, not a new test.

---

## Datasets

| Disease | Studies | Ordered pairs | Source | Already analyzed with geodesic |
|---------|---------|---------------|--------|-------------------------------|
| CRC | 11 | 110 | curatedMetagenomicData | Yes (expansion/results/crc/) |
| T2D | 6 | 30 | curatedMetagenomicData | Yes (expansion/results/t2d/) |

CRC is the primary test. T2D is reported for completeness but is expected null (6 studies, effective N too small for any metric).

---

## What counts as success

1. BN interaction ratio has wider spread (CV) than geodesic on CRC pairs (K1 passes)
2. BN partial rho is positive and exceeds geodesic partial rho (+0.271) on CRC
3. BN clustered bootstrap CI has > 90% mass above zero

If all three hold, the interpretation shifts from "geodesic predicts transportability" to "the classification-relevant component of cross-cohort divergence predicts transportability, and total subspace distance is a poor proxy." This aligns with the construct-validity finding that M1 fails and M3 survives.

## What counts as failure

1. BN CV < geodesic CV → harmonized MetaPhlAn3 lacks sufficient heterogeneity for geometric prediction (any metric)
2. BN partial rho < geodesic partial rho → BN does not improve over the baseline metric; the decomposition adds noise
3. Both BN and DA yield partial rho < +0.10 on CRC → geometric metrics do not predict cross-cohort microbiome classifier degradation at the available sample size

Any of these outcomes is reported honestly. Failure mode 1 is the most informative — it would suggest the JL confound identified in Paper D extends to all geometric metrics on this data type, making "insufficient heterogeneity in harmonized metagenomic profiles" the paper's primary finding.

---

## Relationship to other papers

This addendum positions Paper 1 within the Grassmannian transportability trilogy:

- **Paper 5** (drug-perturbation geometry): DI predicts drug transportability across cell lines
- **Paper D** (construct validity): M1/geodesic fails, M3/direction stability survives
- **Paper 1** (this paper): tests whether classification-relevant metrics (BN, DA) rescue the cohort-transfer prediction that raw geodesic fails

The three-paper arc is: "total subspace distance doesn't predict transport; classification-projected metrics do; here are the boundary conditions."

---

## Code

Analysis script: `expansion/run_bn_da_expansion.py` (to be written after this commit is frozen)

Scorer functions (pre-existing, unmodified):
- `src/transportability.py:bracket_norm_score` (lines 262–332)
- `src/transportability.py:directional_transport_score` (lines 215–259)
- `src/transportability.py:discriminant_direction` (lines 188–212)

---

## Deviations

Any post-hoc additions will be logged below with date and flagged as exploratory.

_(none yet)_

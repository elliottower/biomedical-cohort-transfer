# Geometric Transportability — TODO

## Current state (paper v1d)

Grassmannian geodesic distance detects batch structure (z=27.47) but doesn't
predict cross-cohort classifier degradation on MTBLS7260 (15 same-study plates,
~72 samples each). Three failure modes identified:
1. Target noise: AUC SE ~0.10 on ~0.5 range (~12 positives/plate)
2. Structural mismatch: geodesic is symmetric, AUC gap is directional
3. No dynamic range: same-study plates, CV 3.5%

## Paper v2: directional transport (NEXT)

Turn the null into a positive boundary result: "symmetric geometry fails;
directional, label-aware transport succeeds."

### Predictors
- (a) Directional transport-map projection onto discriminant direction (NEW)
- (b) Bracket-norm batch/label decomposition (NEW)
- (c) Grassmann geodesic (known-to-fail control)
- (d) Centroid distance (naive baseline)

### Dataset 1 (primary): CRC microbiome meta-analysis
- Wirbel et al. 2019 / Thomas et al. 2019 (Nature Medicine)
- 8-10 independent studies, ~1000+ samples, CRC vs healthy binary
- Species-level abundances from MetaPhlAn3, shared feature space
- Zero access barriers (zellerlab/crc_meta on GitHub)
- Folder: `microbiome-crc/`
- Status: **downloading data**

### Dataset 2 (secondary, same paper): metabolomics
- QMDiab (Figshare, T2D, 374 samples, multi-platform) or
- ADNI1 p180 (multi-site AD, 732 samples, free DUA)
- Folder: `metabolomics-multilab/`
- Status: **pending — run after CRC validates the method**

### Method code (DONE)
- [x] directional_transport_score() in src/transportability.py
- [x] bracket_norm_score() in src/transportability.py
- [x] Verified on MTBLS7260: all predictors fail (expected — noise floor)
- [x] Saved 210 ordered-pair results: metabolomics-multilab/mtbls7260_ordered_pairs.csv

### Steps
1. [x] Implement directional transport-map predictor
2. [x] Implement bracket-norm batch/label decomposition
3. [x] Verify on MTBLS7260 (confirms noise is the bottleneck, not method)
4. [x] Download CRC microbiome data (Zenodo 3517209 / Wirbel et al. 2019)
5. [x] Preprocessing: 9 studies, 824 samples, 609 species after 10% filter
6. [x] Run 72 ordered pairs: raw correlations all null (as on MTBLS7260)
7. [x] KEY FINDING: partial correlation (| source AUC) reveals geodesic rho=+0.613
8. [x] Multivariate model: gap ~ int_auc + geodesic, R²=0.693 (+0.226 from geodesic)
9. [x] Null model: z=25.67, studies well-separated from permutation null
10. [x] k-sensitivity: partial rho stable 0.56-0.63 across k=10 to k=50
11. [x] Bootstrap 95% CI: partial rho [+0.439, +0.749], 100% positive
12. [ ] Add metabolomics dataset (QMDiab or ADNI) for cross-domain validation
13. [ ] Write paper_v2a.tex

## Later: additional experiments

- [ ] k-sensitivity deep dive (why RF works at k=15,20 but LR doesn't)
- [ ] Nonlinear embeddings (UMAP/diffusion map subspaces vs PCA)
- [ ] Connection to Ricci curvature (link to causal-inference-neuro paper)
- [ ] Fisher-Rao fix: why it's degenerate on MTBLS7260, conditions for non-degeneracy

## Restyled papers
- [x] geometric-transportability paper_v1d.tex (peer review fixes applied)
- [x] causal-inference-neuro paper_draft_v5b.tex (paragraph headers removed)

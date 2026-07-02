# biomedical-cohort-transfer

Grassmannian geodesic distance predicts cross-cohort classifier degradation in biomedical data, after controlling for source classifier quality.

## What this does

Classifiers trained on one cohort (hospital, study, lab) routinely fail on another. This repo tests whether **geodesic distance between PCA subspaces on the Grassmannian manifold** can predict how much a classifier will degrade — before you train it on the target cohort.

The key insight: raw correlations between geometric distance and classifier degradation are null because the AUC gap conflates source classifier quality with distributional shift. After partialing out source internal AUC, geodesic distance becomes a strong predictor on datasets with genuine analytical heterogeneity.

## Results

| Dataset | Cohorts | Partial rho | Status |
|---|---|---|---|
| CRC microbiome (9 studies, 7 countries) | 72 pairs | +0.61 | Positive |
| QMDiab metabolomics (3 biofluids x 3 ethnicities) | 72 pairs | +0.40 | Positive |
| MTBLS7260 metabolomics (15 plates) | 210 pairs | null | Underpowered |
| IBD metagenomics (5 studies) | 20 pairs | -0.11 | Null (within-cohort variance) |
| SPIROMICS COPD (8 sites) | 56 pairs | +0.08 | Null (centralized platform) |
| Breast cancer GEO (7 Affymetrix studies) | 42 pairs | -0.004 | Null (shared platform) |
| TCGA-BRCA (14 tissue source sites) | 182 pairs | -0.12 | Null (centralized sequencing) |

The method works when cohorts have different measurement processes (sequencing pipelines, biological matrices). It returns null when cohorts share standardized platforms — whether centralized (TCGA, SPIROMICS) or shared across independent sites (Affymetrix HG-U133A).

## Usage

```bash
# Setup
uv sync

# CRC microbiome analysis
PYTHONPATH=. uv run python microbiome-crc/run_crc.py

# QMDiab metabolomics analysis
PYTHONPATH=. uv run python metabolomics-multilab/run_qmdiab.py

# MTBLS7260 metabolomics (original dataset)
PYTHONPATH=. uv run python src/run_multicohort.py

# Batch 2 extensions (IBD, SPIROMICS)
PYTHONPATH=. uv run python batch2_extensions/ibd_microbiome/run_disease.py batch2_extensions/ibd_microbiome/data --disease IBD
PYTHONPATH=. uv run python batch2_extensions/ibd_microbiome/run_disease.py batch2_extensions/metabolomics_multisite/data --disease COPD

# Batch 3 extensions (breast cancer GEO, TCGA, lung cancer)
PYTHONPATH=. uv run python batch3_extensions/breast_cancer_expression/prepare_geo_data.py
PYTHONPATH=. uv run python batch2_extensions/ibd_microbiome/run_disease.py batch3_extensions/breast_cancer_expression/data --disease Breast_Cancer --pre-transformed

# Tests
PYTHONPATH=. uv run python -m pytest tests/ -v
```

## Paper

The current manuscript is `paper_v4.tex`. Compile with:

```bash
pdflatex paper_v4 && bibtex paper_v4 && pdflatex paper_v4 && pdflatex paper_v4
```

## License

MIT

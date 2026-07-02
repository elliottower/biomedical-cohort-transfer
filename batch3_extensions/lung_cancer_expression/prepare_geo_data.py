"""Download and prepare lung cancer gene expression data from GEO.

Downloads multiple Affymetrix HG-U133 Plus 2.0 (GPL570) lung cancer studies
from NCBI GEO. Binary task: tumor tissue vs adjacent normal.

Studies (all GPL570 — Affymetrix HG-U133 Plus 2.0):
  GSE19188  — Hou 2010, Erasmus MC, Netherlands (91 tumor, 65 normal)
  GSE18842  — Sanchez-Palencia 2011, Hospital San Cecilio, Spain (46 tumor, 45 normal)
  GSE32863  — Selamat 2012, UCSF, USA (58 tumor, 58 normal)
  GSE19804  — Lu 2010, National Taiwan Univ (60 tumor, 60 normal)
  GSE27262  — Wei 2012, Taipei Veterans General, Taiwan (25 tumor, 25 normal)
  GSE43458  — Kabbout 2013, MD Anderson, USA (80 tumor, 30 normal)

Usage:
  PYTHONPATH=. uv run python batch3_extensions/lung_cancer_expression/prepare_geo_data.py
"""
import numpy as np
import pandas as pd
import GEOparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

DATA_DIR = Path("batch3_extensions/lung_cancer_expression/data")
GEO_CACHE = DATA_DIR / "geo_cache"
OUT_DIR = DATA_DIR

PLATFORM = "GPL570"

STUDIES = {
    "GSE19188": "Hou_2010_Netherlands",
    "GSE18842": "Sanchez_2011_Spain",
    "GSE32863": "Selamat_2012_USA",
    "GSE19804": "Lu_2010_Taiwan",
    "GSE27262": "Wei_2012_Taiwan",
    "GSE43458": "Kabbout_2013_Houston",
}

TUMOR_KEYWORDS = ["tumor", "tumour", "cancer", "carcinoma", "nsclc",
                   "adenocarcinoma", "squamous", "lung cancer", "malignant"]
NORMAL_KEYWORDS = ["normal", "adjacent normal", "non-tumor", "non-tumour",
                    "nontumor", "healthy", "control", "non-cancerous",
                    "uninvolved", "paired normal"]

MIN_SAMPLES = 15
TOP_PROBES = 5000


def classify_sample(pheno_row):
    """Classify a sample as tumor or normal from its metadata."""
    text_fields = []
    for col in pheno_row.index:
        val = str(pheno_row[col]).lower()
        if any(k in col.lower() for k in ["characteristic", "source", "title", "description"]):
            text_fields.append(val)

    full_text = " ".join(text_fields)

    for kw in NORMAL_KEYWORDS:
        if kw in full_text:
            return "normal"
    for kw in TUMOR_KEYWORDS:
        if kw in full_text:
            return "tumor"
    return None


def load_study(gse_id, study_name):
    print(f"\n[{datetime.now():%H:%M:%S}] Downloading {gse_id} ({study_name})...")
    GEO_CACHE.mkdir(parents=True, exist_ok=True)
    gse = GEOparse.get_GEO(gse_id, destdir=str(GEO_CACHE), silent=True)

    pheno = gse.phenotype_data
    if "platform_id" in pheno.columns:
        gpl_mask = pheno["platform_id"] == PLATFORM
        if gpl_mask.sum() < len(pheno):
            print(f"  Filtering to {PLATFORM}: {gpl_mask.sum()}/{len(pheno)} samples")
            pheno = pheno[gpl_mask]

    if len(pheno) < MIN_SAMPLES:
        print(f"  Skipping: only {len(pheno)} samples on {PLATFORM}")
        return None, None

    labels = pheno.apply(classify_sample, axis=1)
    n_tumor = (labels == "tumor").sum()
    n_normal = (labels == "normal").sum()
    n_unknown = labels.isna().sum()
    print(f"  {len(pheno)} samples: {n_tumor} tumor, {n_normal} normal, {n_unknown} unlabeled")

    if n_tumor < 10 or n_normal < 10:
        print(f"  Skipping: insufficient labeled samples")
        return None, None

    labeled = labels.dropna()
    pheno = pheno.loc[labeled.index]
    sample_ids = pheno.index.tolist()

    expr = gse.pivot_samples("VALUE")
    expr = expr[[s for s in sample_ids if s in expr.columns]]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(how="all")
    print(f"  Expression: {expr.shape[0]} probes x {expr.shape[1]} samples")

    common = [s for s in sample_ids if s in expr.columns]
    meta = pd.DataFrame({
        "Sample_ID": common,
        "Study": study_name,
        "Group": labeled[common].values,
    })

    return expr[common], meta


def main():
    all_expressions = []
    all_meta = []
    loaded_studies = []

    for gse_id, study_name in tqdm(STUDIES.items(), desc="Loading studies"):
        expr, meta = load_study(gse_id, study_name)
        if expr is not None:
            all_expressions.append(expr)
            all_meta.append(meta)
            loaded_studies.append(study_name)

    if len(loaded_studies) < 3:
        print(f"\nOnly {len(loaded_studies)} studies loaded. Need at least 3.")
        return

    print(f"\n[{datetime.now():%H:%M:%S}] Merging {len(loaded_studies)} studies...")

    common_probes = set(all_expressions[0].index)
    for expr in all_expressions[1:]:
        common_probes &= set(expr.index)
    common_probes = sorted(common_probes)
    print(f"  {len(common_probes)} common probes across all studies")

    merged_expr = pd.concat([e.loc[common_probes] for e in all_expressions], axis=1)
    merged_meta = pd.concat(all_meta, axis=0, ignore_index=True)

    probe_var = merged_expr.var(axis=1)
    top_probes = probe_var.nlargest(min(TOP_PROBES, len(probe_var))).index
    merged_expr = merged_expr.loc[top_probes]
    print(f"  Selected top {len(top_probes)} most variable probes")

    print(f"\n  Final: {merged_expr.shape[0]} probes x {merged_expr.shape[1]} samples")
    print(f"  Groups: {merged_meta['Group'].value_counts().to_dict()}")
    for study in loaded_studies:
        sub = merged_meta[merged_meta["Study"] == study]
        n_t = (sub["Group"] == "tumor").sum()
        n_n = (sub["Group"] == "normal").sum()
        print(f"    {study}: {n_t} tumor, {n_n} normal")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_expr.to_csv(OUT_DIR / "species_profiles.tsv", sep="\t")
    merged_meta.to_csv(OUT_DIR / "metadata.tsv", sep="\t", index=False)
    print(f"\n[{datetime.now():%H:%M:%S}] Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

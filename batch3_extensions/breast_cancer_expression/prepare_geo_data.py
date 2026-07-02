"""Download and prepare breast cancer gene expression data from GEO.

Downloads multiple Affymetrix HG-U133A (GPL96) breast cancer studies from NCBI GEO.
Uses clinical ER status when available in GEO metadata; falls back to ESR1 expression
(bimodal GMM) when metadata is absent.

Studies (all GPL96 — Affymetrix HG-U133A):
  GSE2034   — Wang 2005, Erasmus MC Rotterdam, Netherlands (286 samples)
  GSE7390   — Desmedt 2007, Jules Bordet Brussels, Belgium (198 samples)
  GSE1456   — Pawitan 2005, Karolinska Stockholm, Sweden (159 samples on GPL96)
  GSE11121  — Schmidt 2008, University of Mainz, Germany (200 samples)
  GSE4922   — Ivshina 2006, Karolinska/Singapore (289 GPL96, has ER status)
  GSE6532   — Loi 2007, London/Guys/Uppsala (327 GPL96, has ER status)
  GSE25066  — Hatzis 2011, MD Anderson, USA (508 samples)

Usage:
  PYTHONPATH=. uv run python batch3_extensions/breast_cancer_expression/prepare_geo_data.py
"""
import numpy as np
import pandas as pd
import GEOparse
from pathlib import Path
from datetime import datetime
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

DATA_DIR = Path("batch3_extensions/breast_cancer_expression/data")
GEO_CACHE = DATA_DIR / "geo_cache"
OUT_DIR = DATA_DIR

PLATFORM = "GPL96"

STUDIES = {
    "GSE2034": "Wang_2005_Rotterdam",
    "GSE7390": "Desmedt_2007_Brussels",
    "GSE1456": "Pawitan_2005_Stockholm",
    "GSE11121": "Schmidt_2008_Mainz",
    "GSE4922": "Ivshina_2006_Singapore",
    "GSE6532": "Loi_2007_London",
    "GSE25066": "Hatzis_2011_Houston",
}

ESR1_PROBES = ["205225_at", "211233_x_at", "211234_x_at", "211235_s_at",
               "211627_x_at", "215551_at", "215552_s_at"]

ER_FIELD_PATTERNS = {
    "GSE7390": ("characteristics_ch1.12.er", {"1": "ER_positive", "0": "ER_negative"}),
    "GSE4922": ("characteristics_ch1.10.ER status", {"ER+": "ER_positive", "ER-": "ER_negative"}),
    "GSE6532": ("characteristics_ch1.6.er", {"1": "ER_positive", "0": "ER_negative"}),
}

MIN_SAMPLES = 30
TOP_PROBES = 5000


def get_er_from_metadata(pheno, gse_id):
    """Try to extract ER status from GEO phenotype data."""
    if gse_id in ER_FIELD_PATTERNS:
        col, mapping = ER_FIELD_PATTERNS[gse_id]
        if col in pheno.columns:
            er = pheno[col].map(mapping)
            valid = er.dropna()
            if len(valid) >= MIN_SAMPLES:
                n_pos = (valid == "ER_positive").sum()
                n_neg = (valid == "ER_negative").sum()
                print(f"  Clinical ER: {n_pos} ER+, {n_neg} ER- ({len(valid)} annotated)")
                return er
    for col in pheno.columns:
        cl = col.lower()
        if ("er" in cl and ("status" in cl or cl.endswith(".er"))) or "estrogen" in cl:
            vals = pheno[col].dropna().unique()
            mapping = {}
            for v in vals:
                vl = str(v).lower().strip()
                if vl in ["1", "positive", "pos", "er+", "p", "yes"]:
                    mapping[v] = "ER_positive"
                elif vl in ["0", "negative", "neg", "er-", "n", "no"]:
                    mapping[v] = "ER_negative"
            if mapping:
                er = pheno[col].map(mapping)
                valid = er.dropna()
                if len(valid) >= MIN_SAMPLES:
                    n_pos = (valid == "ER_positive").sum()
                    n_neg = (valid == "ER_negative").sum()
                    print(f"  Clinical ER ({col}): {n_pos} ER+, {n_neg} ER-")
                    return er
    return None


def assign_er_from_esr1(expression_df, study_name):
    """Assign ER+/ER- status from ESR1 expression using a 2-component GMM."""
    esr1_probes = [p for p in ESR1_PROBES if p in expression_df.index]
    if not esr1_probes:
        print(f"  WARNING: No ESR1 probes found for {study_name}")
        return None

    esr1_expr = expression_df.loc[esr1_probes].max(axis=0).values.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(esr1_expr)
    labels = gmm.predict(esr1_expr)

    mean0, mean1 = gmm.means_[0, 0], gmm.means_[1, 0]
    if mean0 > mean1:
        er_labels = np.where(labels == 0, "ER_positive", "ER_negative")
    else:
        er_labels = np.where(labels == 1, "ER_positive", "ER_negative")

    n_pos = (er_labels == "ER_positive").sum()
    n_neg = (er_labels == "ER_negative").sum()
    sep = abs(mean0 - mean1) / np.sqrt((gmm.covariances_[0, 0, 0] + gmm.covariances_[1, 0, 0]) / 2)
    print(f"  ESR1 GMM: {n_pos} ER+, {n_neg} ER-, separation={sep:.2f}")
    return pd.Series(er_labels, index=expression_df.columns)


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

    sample_ids = pheno.index.tolist()
    print(f"  {len(sample_ids)} samples on {PLATFORM}")

    expr = gse.pivot_samples("VALUE")
    expr = expr[[s for s in sample_ids if s in expr.columns]]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(how="all")
    print(f"  Expression: {expr.shape[0]} probes x {expr.shape[1]} samples")

    er_labels = get_er_from_metadata(pheno, gse_id)
    er_source = "clinical"
    if er_labels is None:
        print(f"  No clinical ER metadata, using ESR1 GMM...")
        er_labels = assign_er_from_esr1(expr, study_name)
        er_source = "ESR1_GMM"

    if er_labels is None:
        return None, None

    common_samples = [s for s in expr.columns if s in er_labels.index and pd.notna(er_labels[s])]
    if len(common_samples) < MIN_SAMPLES:
        print(f"  Skipping: only {len(common_samples)} samples with ER labels")
        return None, None

    expr = expr[common_samples]
    er = er_labels[common_samples]

    meta = pd.DataFrame({
        "Sample_ID": common_samples,
        "Study": study_name,
        "Group": er.values,
        "ER_source": er_source,
    })

    n_pos = (meta["Group"] == "ER_positive").sum()
    n_neg = (meta["Group"] == "ER_negative").sum()
    if n_pos < 10 or n_neg < 10:
        print(f"  Skipping: too few in one class ({n_pos} ER+, {n_neg} ER-)")
        return None, None

    esr1_in_data = [p for p in ESR1_PROBES if p in expr.index]
    if er_source == "ESR1_GMM":
        expr = expr.drop(index=esr1_in_data, errors="ignore")
        print(f"  Removed {len(esr1_in_data)} ESR1 probes (used for labeling)")

    return expr, meta


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
        n_pos = (sub["Group"] == "ER_positive").sum()
        n_neg = (sub["Group"] == "ER_negative").sum()
        src = sub["ER_source"].iloc[0]
        print(f"    {study}: {n_pos} ER+, {n_neg} ER- (ER from {src})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_expr.to_csv(OUT_DIR / "species_profiles.tsv", sep="\t")
    merged_meta[["Sample_ID", "Study", "Group"]].to_csv(OUT_DIR / "metadata.tsv", sep="\t", index=False)
    print(f"\n[{datetime.now():%H:%M:%S}] Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

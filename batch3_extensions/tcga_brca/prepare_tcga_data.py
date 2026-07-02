"""Download and prepare TCGA-BRCA gene expression data.

Uses tissue source sites (TSS) as cohorts to test transportability of
ER+ vs ER- classification across different hospitals/institutions.

Clinical data from cBioPortal API (ER status, patient IDs).
Expression data from UCSC Xena (pre-compiled gene x sample matrix).

TCGA-BRCA has ~1,100 primary tumor samples from 20+ tissue source sites,
sequenced at centralized genome centers (Broad, Baylor, WashU) but collected
at independent hospitals with different patient populations.

This tests whether the geodesic distance method detects population-level
distributional shift even when analytical processing is centralized.

Usage:
  PYTHONPATH=. uv run python batch3_extensions/tcga_brca/prepare_tcga_data.py
"""
import gzip
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

DATA_DIR = Path("batch3_extensions/tcga_brca/data")
OUT_DIR = DATA_DIR

CBIO_BASE = "https://www.cbioportal.org/api"
STUDY_ID = "brca_tcga"

XENA_EXPR_URL = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2.gz"

MIN_SITE_SAMPLES = 25
TOP_GENES = 5000

TSS_NAMES = {
    "A1": "UNC", "A2": "Christiana", "A7": "Christiana",
    "A8": "Asterand", "AC": "IGC", "AN": "MD_Anderson",
    "AO": "Asterand", "AQ": "ILSBio", "AR": "MD_Anderson",
    "B6": "Henry_Ford", "BH": "Pittsburgh", "C8": "WashU",
    "D8": "Indivumed", "E2": "Emory", "E9": "Indivumed",
    "EW": "ILSBio", "GI": "Poland", "GM": "Miami",
    "HN": "GOG", "JL": "Hopkins", "LL": "Pittsburgh",
    "LQ": "Hawaii", "MS": "Cureline", "OL": "Kansas",
    "PE": "Barretos", "PL": "Ottawa", "S3": "Pittsburgh",
}


def download_clinical_data():
    print(f"[{datetime.now():%H:%M:%S}] Downloading clinical data from cBioPortal...")
    url = f"{CBIO_BASE}/studies/{STUDY_ID}/clinical-data"
    params = {"clinicalDataType": "PATIENT", "projection": "DETAILED"}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    records = {}
    for item in data:
        pid = item["patientId"]
        if pid not in records:
            records[pid] = {}
        records[pid][item["clinicalAttributeId"]] = item["value"]

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "PATIENT_ID"
    print(f"  {len(df)} patients, {len(df.columns)} attributes")
    return df


def download_expression_xena():
    """Download pre-compiled expression matrix from UCSC Xena."""
    cache = DATA_DIR / "xena_HiSeqV2.tsv.gz"
    if cache.exists():
        print(f"[{datetime.now():%H:%M:%S}] Loading cached Xena expression...")
    else:
        print(f"[{datetime.now():%H:%M:%S}] Downloading expression from UCSC Xena (~64 MB)...")
        resp = requests.get(XENA_EXPR_URL, timeout=300, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with open(cache, "wb") as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
        print()

    print(f"  Parsing expression matrix...")
    expr = pd.read_csv(cache, sep="\t", index_col=0, compression="gzip")
    print(f"  {expr.shape[0]} genes x {expr.shape[1]} samples")
    return expr


def get_tss_from_barcode(barcode):
    parts = str(barcode).split("-")
    if len(parts) >= 2:
        return parts[1]
    return None


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    clinical_cache = DATA_DIR / "clinical_patient.json"
    if clinical_cache.exists():
        print(f"[{datetime.now():%H:%M:%S}] Loading cached clinical data...")
        clinical = pd.read_json(clinical_cache)
    else:
        clinical = download_clinical_data()
        clinical.to_json(clinical_cache)

    er_col = None
    for candidate in ["ER_STATUS_BY_IHC", "ER_IHC"]:
        if candidate in clinical.columns:
            er_col = candidate
            break
    if er_col is None:
        er_candidates = [c for c in clinical.columns if "ER" in c.upper()]
        er_col = er_candidates[0] if er_candidates else None
    if er_col is None:
        print("  ERROR: No ER status column found")
        return

    print(f"  ER column: {er_col} -> {clinical[er_col].value_counts().to_dict()}")

    er_map = {}
    for val in clinical[er_col].unique():
        vl = str(val).lower().strip()
        if vl in ["positive", "pos", "1", "yes"]:
            er_map[val] = "ER_positive"
        elif vl in ["negative", "neg", "0", "no"]:
            er_map[val] = "ER_negative"

    clinical["Group"] = clinical[er_col].map(er_map)
    clinical = clinical.dropna(subset=["Group"])
    print(f"  {len(clinical)} patients with ER status")

    clinical["TSS"] = clinical.index.map(get_tss_from_barcode)
    clinical = clinical.dropna(subset=["TSS"])

    tss_counts = clinical["TSS"].value_counts()
    good_tss = tss_counts[tss_counts >= MIN_SITE_SAMPLES].index
    clinical = clinical[clinical["TSS"].isin(good_tss)]

    print(f"\n  Sites with >= {MIN_SITE_SAMPLES} samples:")
    for tss in sorted(good_tss):
        sub = clinical[clinical["TSS"] == tss]
        name = TSS_NAMES.get(tss, "Unknown")
        n_pos = (sub["Group"] == "ER_positive").sum()
        n_neg = (sub["Group"] == "ER_negative").sum()
        print(f"    {tss} ({name}): {n_pos} ER+, {n_neg} ER- ({len(sub)} total)")

    expr = download_expression_xena()

    sample_map = {}
    for pid in clinical.index:
        for suffix in ["-01", "-01A", "-01B", "-01C"]:
            sid = f"{pid}{suffix}"
            if sid in expr.columns:
                sample_map[pid] = sid
                break

    matched_pids = [pid for pid in clinical.index if pid in sample_map]
    print(f"\n  {len(matched_pids)} / {len(clinical)} patients matched to expression data")

    clinical = clinical.loc[matched_pids]

    meta_rows = []
    expr_cols = []
    for pid in clinical.index:
        sid = sample_map[pid]
        tss = clinical.loc[pid, "TSS"]
        name = TSS_NAMES.get(tss, tss)
        meta_rows.append({
            "Sample_ID": sid,
            "Study": f"{tss}_{name}",
            "Group": clinical.loc[pid, "Group"],
        })
        expr_cols.append(sid)

    meta_df = pd.DataFrame(meta_rows)
    expr = expr[expr_cols]

    zero_var = expr.var(axis=1) == 0
    expr = expr[~zero_var]
    gene_var = expr.var(axis=1)
    top_genes = gene_var.nlargest(min(TOP_GENES, len(gene_var))).index
    expr = expr.loc[top_genes]
    print(f"  Selected top {len(top_genes)} most variable genes")

    tss_filter = meta_df["Study"].value_counts()
    keep_tss = tss_filter[tss_filter >= MIN_SITE_SAMPLES].index
    mask = meta_df["Study"].isin(keep_tss)
    meta_df = meta_df[mask]
    expr = expr[meta_df["Sample_ID"].tolist()]

    print(f"\n  Final: {expr.shape[0]} genes x {expr.shape[1]} samples")
    for study in sorted(meta_df["Study"].unique()):
        sub = meta_df[meta_df["Study"] == study]
        n_pos = (sub["Group"] == "ER_positive").sum()
        n_neg = (sub["Group"] == "ER_negative").sum()
        print(f"    {study}: {n_pos} ER+, {n_neg} ER- ({len(sub)} total)")
    print(f"  Groups: {meta_df['Group'].value_counts().to_dict()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expr.to_csv(OUT_DIR / "species_profiles.tsv", sep="\t")
    meta_df.to_csv(OUT_DIR / "metadata.tsv", sep="\t", index=False)
    print(f"\n[{datetime.now():%H:%M:%S}] Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

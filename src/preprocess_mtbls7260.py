"""Preprocess MTBLS7260 (pancreatic cancer, PLCO trial, 10 sites).

Extracts feature matrices from MAF files, maps sample IDs to clinical
labels and batch/plate metadata, and creates cohort splits for
transportability testing.

Usage:
    python src/preprocess_mtbls7260.py
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


RAW_DIR = Path("data/raw/MTBLS7260")
OUT_DIR = Path("data/processed")


def extract_sample_metadata():
    """Parse sample table to get sample_id -> disease label mapping."""
    print(f"[{datetime.now():%H:%M:%S}] Parsing sample metadata...")

    s = pd.read_csv(RAW_DIR / "s_MTBLS7260.txt", sep="\t")

    s = s[~s["Factor Value[Disease]"].str.contains("Excluded", na=False)]

    label_map = {"Healthy Control": 0, "Cancer": 1}
    s["label"] = s["Factor Value[Disease]"].map(label_map)

    meta = s[["Source Name", "Sample Name", "label"]].copy()
    meta.columns = ["source_name", "sample_id", "label"]

    print(f"  {len(meta)} samples after excluding flagged")
    print(f"  Labels: {meta['label'].value_counts().to_dict()}")
    return meta


def extract_feature_matrix(maf_path, meta):
    """Extract samples x features matrix from a MAF file.

    MAF files have metabolite metadata in the first ~21 columns,
    then sample abundances in remaining columns. Column names encode
    sample IDs and batch/plate info.
    """
    print(f"[{datetime.now():%H:%M:%S}] Parsing {maf_path.name}...")

    maf = pd.read_csv(maf_path, sep="\t")

    metadata_cols = [
        "database_identifier", "chemical_formula", "smiles", "inchi",
        "metabolite_identification", "mass_to_charge", "fragmentation",
        "modifications", "charge", "retention_time", "taxid", "species",
        "database", "database_version", "reliability", "uri",
        "search_engine", "search_engine_score",
        "smallmolecule_abundance_sub", "smallmolecule_abundance_stdev_sub",
        "smallmolecule_abundance_std_error_sub",
    ]
    sample_cols = [c for c in maf.columns if c not in metadata_cols]

    qc_pattern = re.compile(r"(PQC|HQC|LQC|PreH|MSe|Blank|blank)", re.IGNORECASE)
    sample_cols = [c for c in sample_cols if not qc_pattern.search(c)]

    print(f"  {len(sample_cols)} sample columns (after removing QC)")
    print(f"  {len(maf)} metabolite features")

    feature_names = []
    seen = {}
    for _, row in maf.iterrows():
        name = row.get("metabolite_identification", "")
        mz = row.get("mass_to_charge", "")
        rt = row.get("retention_time", "")
        if pd.notna(name) and str(name).strip():
            base = str(name).strip()
        else:
            base = f"mz{mz}_rt{rt}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        feature_names.append(f"{base}__{count}" if count > 0 else base)

    X_raw = maf[sample_cols].T
    X_raw.columns = feature_names
    X_raw.index.name = "raw_col"
    X_raw = X_raw.reset_index()

    sample_id_pattern = re.compile(r"[A-Z]{4}\d{4}")
    plate_pattern = re.compile(r"(PL\d+)")

    ids = []
    plates = []
    for col in X_raw["raw_col"]:
        m = sample_id_pattern.search(col)
        ids.append(m.group() if m else col)
        p = plate_pattern.search(col)
        plates.append(p.group() if p else "unknown")

    X_raw["sample_id"] = ids
    X_raw["plate"] = plates

    merged = X_raw.merge(meta, on="sample_id", how="inner")
    print(f"  {len(merged)} samples matched to metadata")

    feature_cols = feature_names
    X = merged[feature_cols].values.astype(np.float64)
    labels = merged["label"].values.astype(int)
    plates_arr = merged["plate"].values
    sample_ids = merged["sample_id"].values

    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  Imputing {nan_count} missing values (column median)")
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_medians[j]

    return X, labels, plates_arr, sample_ids, feature_cols


def create_cohort_splits(X, labels, plates, sample_ids, feature_names):
    """Split into cohorts based on batch/plate structure.

    Strategy: group plates into two cohorts (roughly balanced) to
    create a natural cross-batch transportability test.
    """
    print(f"\n[{datetime.now():%H:%M:%S}] Creating cohort splits...")

    unique_plates = sorted(set(plates))
    print(f"  Plates: {unique_plates}")

    plate_counts = {}
    for p in unique_plates:
        mask = plates == p
        n = mask.sum()
        n_cancer = labels[mask].sum()
        plate_counts[p] = {"n": n, "cancer": int(n_cancer), "control": int(n - n_cancer)}
        print(f"    {p}: {n} samples ({n_cancer} cancer, {n - n_cancer} control)")

    mid = len(unique_plates) // 2
    plates_c1 = set(unique_plates[:mid])
    plates_c2 = set(unique_plates[mid:])

    mask_c1 = np.array([p in plates_c1 for p in plates])
    mask_c2 = np.array([p in plates_c2 for p in plates])

    print(f"\n  Cohort 1 (plates {plates_c1}): {mask_c1.sum()} samples, "
          f"{labels[mask_c1].sum()} cancer")
    print(f"  Cohort 2 (plates {plates_c2}): {mask_c2.sum()} samples, "
          f"{labels[mask_c2].sum()} cancer")

    df1 = pd.DataFrame(X[mask_c1], columns=feature_names)
    df1["label"] = labels[mask_c1]
    df1["sample_id"] = sample_ids[mask_c1]
    df1["plate"] = plates[mask_c1]

    df2 = pd.DataFrame(X[mask_c2], columns=feature_names)
    df2["label"] = labels[mask_c2]
    df2["sample_id"] = sample_ids[mask_c2]
    df2["plate"] = plates[mask_c2]

    return df1, df2


def run():
    meta = extract_sample_metadata()

    maf_files = sorted(RAW_DIR.glob("m_*_v2_maf.tsv"))
    print(f"\n  Found {len(maf_files)} MAF files")

    best_maf = None
    best_n = 0
    for maf_path in maf_files:
        X, labels, plates, sids, fnames = extract_feature_matrix(maf_path, meta)
        if len(labels) > best_n:
            best_n = len(labels)
            best_maf = (X, labels, plates, sids, fnames, maf_path.name)

    if best_maf is None:
        print("  ERROR: No MAF files with matched samples found")
        return

    X, labels, plates, sids, fnames, maf_name = best_maf
    print(f"\n  Using {maf_name} ({len(labels)} samples, {len(fnames)} features)")

    df1, df2 = create_cohort_splits(X, labels, plates, sids, fnames)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df1.to_csv(OUT_DIR / "cohort1.csv", index=False)
    df2.to_csv(OUT_DIR / "cohort2.csv", index=False)

    print(f"\n[{datetime.now():%H:%M:%S}] Saved to {OUT_DIR}/")
    print(f"  cohort1.csv: {df1.shape}")
    print(f"  cohort2.csv: {df2.shape}")


if __name__ == "__main__":
    run()

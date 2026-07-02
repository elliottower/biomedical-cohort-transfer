"""Prepare IBD data from SIAMCAT Zenodo archive for the transportability pipeline.

Reads per-study mOTU abundance tables and metadata from the SIAMCAT archive,
filters to IBD-relevant studies (CD/UC vs CTR), merges into a single abundance
matrix and metadata file.

Usage:
  PYTHONPATH=. uv run python batch2_extensions/ibd_microbiome/prepare_siamcat_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("batch2_extensions/ibd_microbiome/data/siamcat_raw")
OUT_DIR = Path("batch2_extensions/ibd_microbiome/data")

IBD_STUDIES = ["Franzosa_2019", "He_2017", "HMP2", "Lewis_2015", "metaHIT"]
IBD_GROUPS = {"CD", "UC"}
CTR_GROUP = "CTR"


def load_study(study_name):
    motus_file = RAW_DIR / f"{study_name}_motus.tsv"
    meta_file = RAW_DIR / f"meta_{study_name}.tsv"

    print(f"[{datetime.now():%H:%M:%S}] Loading {study_name}...")
    meta = pd.read_csv(meta_file, sep="\t")

    sample_col = "Sample_ID" if "Sample_ID" in meta.columns else meta.columns[0]
    meta = meta.set_index(sample_col)

    ibd_mask = meta["Group"].isin(IBD_GROUPS)
    ctr_mask = meta["Group"] == CTR_GROUP
    keep = ibd_mask | ctr_mask
    meta = meta[keep].copy()

    if "Individual_ID" in meta.columns and "Timepoint" in meta.columns:
        n_before = len(meta)
        rng = np.random.default_rng(42)
        keep_samples = []
        for indiv, sub in meta.groupby("Individual_ID"):
            idx = rng.integers(len(sub))
            keep_samples.append(sub.index[idx])
        meta = meta.loc[keep_samples]
        print(f"  Subsampled {n_before} -> {len(meta)} (one per individual)")

    meta["IBD_label"] = meta["Group"].apply(lambda g: "IBD" if g in IBD_GROUPS else "control")
    meta["Study"] = study_name

    n_ibd = (meta["IBD_label"] == "IBD").sum()
    n_ctr = (meta["IBD_label"] == "control").sum()
    group_counts = {g: int((meta["Group"] == g).sum()) for g in IBD_GROUPS if (meta["Group"] == g).sum() > 0}
    group_str = ", ".join(f"{g}:{c}" for g, c in group_counts.items())
    print(f"  {study_name}: {n_ibd} IBD ({group_str}), {n_ctr} CTR")

    abundance = pd.read_csv(motus_file, sep="\t", index_col=0)
    drop_rows = [idx for idx in abundance.index if str(idx).strip() == "-1"]
    if drop_rows:
        abundance = abundance.drop(drop_rows)

    common = abundance.columns.intersection(meta.index)
    abundance = abundance[common]
    meta = meta.loc[common]

    print(f"  {len(common)} samples matched, {abundance.shape[0]} species")
    return abundance, meta


def main():
    all_abundance = []
    all_meta = []

    for study in IBD_STUDIES:
        abundance, meta = load_study(study)
        all_abundance.append(abundance)
        all_meta.append(meta[["Study", "Group", "IBD_label"]])

    print(f"\n[{datetime.now():%H:%M:%S}] Merging {len(IBD_STUDIES)} studies...")

    merged_abundance = pd.concat(all_abundance, axis=1, join="outer").fillna(0)
    merged_meta = pd.concat(all_meta, axis=0)

    col_sums = merged_abundance.sum(axis=0)
    zero_samples = (col_sums == 0).sum()
    if zero_samples > 0:
        print(f"  Removing {zero_samples} samples with zero total counts")
        merged_abundance = merged_abundance.loc[:, col_sums > 0]
        merged_meta = merged_meta.loc[merged_abundance.columns]

    nonzero_frac = (merged_abundance > 0).mean(axis=1)
    keep_species = nonzero_frac >= 0.01
    print(f"  {keep_species.sum()} species with >=1% prevalence (dropped {(~keep_species).sum()})")
    merged_abundance = merged_abundance[keep_species]

    print(f"\n  Final: {merged_abundance.shape[0]} species x {merged_abundance.shape[1]} samples")
    print(f"  Studies: {merged_meta['Study'].value_counts().to_dict()}")
    print(f"  Groups: {merged_meta['IBD_label'].value_counts().to_dict()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_abundance.to_csv(OUT_DIR / "species_profiles.tsv", sep="\t")
    meta_out = merged_meta.reset_index().rename(columns={merged_meta.index.name or "index": "Sample_ID"})
    meta_out["Group"] = meta_out["IBD_label"]
    meta_out = meta_out[["Sample_ID", "Study", "Group"]]
    meta_out.to_csv(OUT_DIR / "metadata.tsv", sep="\t", index=False)

    print(f"\n[{datetime.now():%H:%M:%S}] Saved to {OUT_DIR}/species_profiles.tsv and metadata.tsv")


if __name__ == "__main__":
    main()

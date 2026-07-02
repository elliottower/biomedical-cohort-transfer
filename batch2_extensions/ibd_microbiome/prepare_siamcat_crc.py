"""Prepare CRC data from SIAMCAT Zenodo archive as a positive control.

Uses the same mOTU profiling pipeline as IBD to validate that the geodesic
distance method works with SIAMCAT-formatted data (confirming the IBD null
result is real, not a bug).

Usage:
  PYTHONPATH=. uv run python batch2_extensions/ibd_microbiome/prepare_siamcat_crc.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("batch2_extensions/ibd_microbiome/data/siamcat_raw")
OUT_DIR = Path("batch2_extensions/ibd_microbiome/data_crc")

CRC_STUDIES = ["Feng_2015", "Thomas_2019", "Vogtmann_2016", "Wirbel_2019",
               "Yachida_2019", "Yu_2017"]
DISEASE_GROUP = "CRC"
CTR_GROUP = "CTR"


def load_study(study_name):
    motus_file = RAW_DIR / f"{study_name}_motus.tsv"
    meta_file = RAW_DIR / f"meta_{study_name}.tsv"

    print(f"[{datetime.now():%H:%M:%S}] Loading {study_name}...")
    meta = pd.read_csv(meta_file, sep="\t")
    sample_col = "Sample_ID" if "Sample_ID" in meta.columns else meta.columns[0]
    meta = meta.set_index(sample_col)

    keep = meta["Group"].isin([DISEASE_GROUP, CTR_GROUP])
    meta = meta[keep].copy()
    meta["Group"] = meta["Group"].map({DISEASE_GROUP: "CRC", CTR_GROUP: "control"})
    meta["Study"] = study_name

    n_crc = (meta["Group"] == "CRC").sum()
    n_ctr = (meta["Group"] == "control").sum()
    print(f"  {study_name}: {n_crc} CRC, {n_ctr} CTR")

    abundance = pd.read_csv(motus_file, sep="\t", index_col=0)
    drop_rows = [idx for idx in abundance.index if str(idx).strip() == "-1"]
    if drop_rows:
        abundance = abundance.drop(drop_rows)

    common = abundance.columns.intersection(meta.index)
    abundance = abundance[common]
    meta = meta.loc[common]
    return abundance, meta


def main():
    all_abundance = []
    all_meta = []

    for study in CRC_STUDIES:
        abundance, meta = load_study(study)
        all_abundance.append(abundance)
        all_meta.append(meta[["Study", "Group"]])

    print(f"\n[{datetime.now():%H:%M:%S}] Merging {len(CRC_STUDIES)} studies...")
    merged_abundance = pd.concat(all_abundance, axis=1, join="outer").fillna(0)
    merged_meta = pd.concat(all_meta, axis=0)

    col_sums = merged_abundance.sum(axis=0)
    zero_samples = (col_sums == 0).sum()
    if zero_samples > 0:
        print(f"  Removing {zero_samples} zero-count samples")
        merged_abundance = merged_abundance.loc[:, col_sums > 0]
        merged_meta = merged_meta.loc[merged_abundance.columns]

    nonzero_frac = (merged_abundance > 0).mean(axis=1)
    keep_species = nonzero_frac >= 0.01
    print(f"  {keep_species.sum()} species >= 1% prevalence")
    merged_abundance = merged_abundance[keep_species]

    print(f"\n  Final: {merged_abundance.shape[0]} species x {merged_abundance.shape[1]} samples")
    print(f"  Studies: {merged_meta['Study'].value_counts().to_dict()}")
    print(f"  Groups: {merged_meta['Group'].value_counts().to_dict()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_abundance.to_csv(OUT_DIR / "species_profiles.tsv", sep="\t")
    meta_out = merged_meta.reset_index().rename(columns={merged_meta.index.name or "index": "Sample_ID"})
    meta_out.to_csv(OUT_DIR / "metadata.tsv", sep="\t", index=False)
    print(f"\n[{datetime.now():%H:%M:%S}] Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

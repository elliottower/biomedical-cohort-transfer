"""Prepare SPIROMICS COPD metabolomics data for transportability analysis.

Downloads metabolite abundance data and metadata from Metabolomics Workbench
REST API (study ST002088), merges across 4 analyses, and exports as TSV files
for the transportability pipeline.

Sites: 7 clinical centers (Columbia, Johns Hopkins, UCLA, UCSF, U of Michigan,
U of Utah, Wake Forest University) + "Other"

Task: Current Smoker vs Former Smoker classification (215 vs 369 samples)

Usage:
  PYTHONPATH=. uv run python batch2_extensions/metabolomics_multisite/prepare_spiromics.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("batch2_extensions/metabolomics_multisite/data/spiromics")
OUT_DIR = Path("batch2_extensions/metabolomics_multisite/data")

DISEASE_LABEL = "Current Smoker"
CONTROL_LABEL = "Former Smoker"
MIN_SITE_SAMPLES = 30


def load_metabolite_data():
    print(f"[{datetime.now():%H:%M:%S}] Loading metabolite data...")

    with open(DATA_DIR / "study_data.json") as f:
        data = json.load(f)

    metabolite_rows = {}
    for entry in data.values():
        name = entry["metabolite_name"]
        if name not in metabolite_rows:
            metabolite_rows[name] = {}
        for sample_id, value in entry["DATA"].items():
            if value is None or value == "":
                continue
            val = float(value)
            if name in metabolite_rows and sample_id in metabolite_rows[name]:
                metabolite_rows[name][sample_id] = max(metabolite_rows[name][sample_id], val)
            else:
                metabolite_rows[name][sample_id] = val

    abundance = pd.DataFrame(metabolite_rows).T
    abundance = abundance.fillna(0)
    print(f"  {abundance.shape[0]} metabolites x {abundance.shape[1]} samples")
    return abundance


def load_metadata():
    print(f"[{datetime.now():%H:%M:%S}] Loading metadata...")

    with open(DATA_DIR / "factors.json") as f:
        factors = json.load(f)

    samples = []
    for entry in factors.values():
        sid = entry["local_sample_id"]
        factor_str = entry.get("factors", "")
        parsed = {"Sample_ID": sid}
        for pair in factor_str.split(" | "):
            if ":" in pair:
                k, v = pair.split(":", 1)
                parsed[k.strip()] = v.strip()
        samples.append(parsed)

    meta = pd.DataFrame(samples)
    meta = meta.set_index("Sample_ID")
    print(f"  {len(meta)} samples, columns: {list(meta.columns)}")
    return meta


def main():
    abundance = load_metabolite_data()
    meta = load_metadata()

    keep = meta["smokStatus"].isin([DISEASE_LABEL, CONTROL_LABEL])
    meta = meta[keep].copy()
    meta["Group"] = meta["smokStatus"].map({DISEASE_LABEL: "disease", CONTROL_LABEL: "control"})
    meta["Study"] = meta["SITE_cat"]
    print(f"\n  {len(meta)} samples after filtering to {DISEASE_LABEL} vs {CONTROL_LABEL}")

    common = abundance.columns.intersection(meta.index)
    abundance = abundance[common]
    meta = meta.loc[common]
    print(f"  {len(common)} samples with metabolite data")

    site_counts = meta["Study"].value_counts()
    good_sites = site_counts[site_counts >= MIN_SITE_SAMPLES].index
    meta = meta[meta["Study"].isin(good_sites)]
    abundance = abundance[meta.index]
    print(f"\n  Sites with >= {MIN_SITE_SAMPLES} samples:")
    for site in sorted(good_sites):
        sub = meta[meta["Study"] == site]
        n_dis = (sub["Group"] == "disease").sum()
        n_ctr = (sub["Group"] == "control").sum()
        print(f"    {site}: {n_dis} current, {n_ctr} former ({len(sub)} total)")

    zero_cols = (abundance == 0).all(axis=0)
    if zero_cols.any():
        abundance = abundance.loc[:, ~zero_cols]
        meta = meta.loc[abundance.columns]

    nonzero_frac = (abundance > 0).mean(axis=1)
    keep_features = nonzero_frac >= 0.5
    abundance = abundance[keep_features]
    print(f"\n  {keep_features.sum()} metabolites with >= 50% prevalence")

    print(f"\n  Final: {abundance.shape[0]} metabolites x {abundance.shape[1]} samples")
    print(f"  Groups: {meta['Group'].value_counts().to_dict()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    abundance.to_csv(OUT_DIR / "species_profiles.tsv", sep="\t")
    meta_reset = meta.reset_index()
    meta_reset = meta_reset.rename(columns={meta_reset.columns[0]: "Sample_ID"})
    meta_out = meta_reset[["Sample_ID", "Study", "Group"]]
    meta_out.to_csv(OUT_DIR / "metadata.tsv", sep="\t", index=False)
    print(f"\n[{datetime.now():%H:%M:%S}] Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

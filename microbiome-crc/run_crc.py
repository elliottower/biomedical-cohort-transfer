"""Cross-cohort transportability on CRC microbiome meta-analysis.

Tests directional transport predictor vs symmetric baselines on ordered
study pairs from Wirbel et al. 2019 / Thomas et al. 2019.

Usage: PYTHONPATH=. uv run python microbiome-crc/run_crc.py
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from tqdm import tqdm

from src.transportability import (
    directional_transport_score, bracket_norm_score,
    geodesic_distance, top_k_subspace, centroid_distance,
    domain_classifier_auc, variance_ratio,
)
from src.external_validation import external_validation, internal_validation
from src.embed import embed

DATA_DIR = Path("microbiome-crc/data")
RESULTS_DIR = Path("microbiome-crc/results")
K = 10
SEED = 42


def load_crc_cohorts(min_prevalence=0.1):
    """Load CRC abundance tables and metadata, one cohort per study.

    Data from Wirbel et al. 2019 / zellerlab/crc_meta (Zenodo).
    Species profiles are (species x samples), metadata is (samples x fields).

    Returns dict: study_name -> {"X": array, "y": array, "n": int, ...}
    """
    print(f"[{datetime.now():%H:%M:%S}] Loading CRC cohorts...")

    species_file = DATA_DIR / "species_profiles.tsv"
    meta_file = DATA_DIR / "meta_all.tsv"

    if not species_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            f"Download CRC data first. Run:\n"
            f"  cd microbiome-crc && bash download.sh"
        )

    meta = pd.read_csv(meta_file, sep="\t")
    crc_studies = [s for s in meta["Study"].unique() if "CRC" in s]
    meta_crc = meta[meta["Study"].isin(crc_studies) & meta["Group"].isin(["CTR", "CRC"])].copy()
    meta_crc = meta_crc.set_index("Sample_ID")

    feat = pd.read_csv(species_file, sep="\t", index_col=0)
    common = feat.columns.intersection(meta_crc.index)
    feat = feat[common].T
    meta_crc = meta_crc.loc[common]

    prevalence = (feat > 0).mean(axis=0)
    keep_features = prevalence >= min_prevalence
    feat = feat.loc[:, keep_features]
    print(f"  {keep_features.sum()} features after {min_prevalence:.0%} prevalence filter "
          f"(dropped {(~keep_features).sum()})")

    feat_rel = feat.div(feat.sum(axis=1), axis=0)
    X_all = np.log1p(feat_rel.values.astype(np.float64) * 1e6)

    y_all = (meta_crc["Group"] == "CRC").astype(int).values

    studies = sorted(meta_crc["Study"].unique())
    cohorts = {}
    for study in studies:
        mask = (meta_crc["Study"] == study).values
        X = X_all[mask]
        y = y_all[mask]
        if len(y) < 20 or len(np.unique(y)) < 2:
            print(f"  Skipping {study}: {len(y)} samples")
            continue
        cohorts[study] = {
            "X": X, "y": y, "n": len(y),
            "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        }
        print(f"  {study}: {len(y)} samples ({y.sum():.0f} CRC, {(1-y).sum():.0f} healthy)")

    print(f"  {len(cohorts)} studies, {sum(c['n'] for c in cohorts.values())} total samples")
    return cohorts


def run_ordered_pairs(cohorts, k=K):
    """Run all ordered pairs: directional + symmetric predictors vs AUC gap."""
    print(f"\n[{datetime.now():%H:%M:%S}] Running ordered pairs...")

    studies = sorted(cohorts.keys())
    n = len(studies)
    n_pairs = n * (n - 1)

    rows = []
    for i in tqdm(range(n), desc="Ordered pairs"):
        for j in range(n):
            if i == j:
                continue
            s1, s2 = studies[i], studies[j]
            X1, y1 = cohorts[s1]["X"], cohorts[s1]["y"]
            X2, y2 = cohorts[s2]["X"], cohorts[s2]["y"]

            dt = directional_transport_score(X1, y1, X2, y2, k=k)
            bn = bracket_norm_score(X1, y1, X2, y2)

            U1, _ = top_k_subspace(X1, k)
            U2, _ = top_k_subspace(X2, k)
            gdist = geodesic_distance(U1, U2)
            cdist = centroid_distance(X1, X2)

            ext = external_validation(X1, y1, X2, y2, n_bootstrap=0, seed=SEED)
            auc_int = internal_validation(X1, y1, seed=SEED)
            auc_gap = auc_int["mean_auc"] - ext["forward"]["auc"]

            rows.append({
                "from": s1,
                "to": s2,
                "n_from": len(y1),
                "n_to": len(y2),
                "auc_gap": auc_gap,
                "ext_auc": ext["forward"]["auc"],
                "int_auc": auc_int["mean_auc"],
                "dt_score": dt["score"],
                "dt_abs": dt["abs_score"],
                "dt_alignment": dt["discriminant_alignment"],
                "bn_score": bn["score"],
                "bn_batch": bn["batch_norm"],
                "bn_interaction": bn["interaction_norm"],
                "geodesic": gdist,
                "centroid_dist": cdist,
            })

    return pd.DataFrame(rows)


def print_correlations(df):
    """Print Spearman correlations between all predictors and AUC gap."""
    print(f"\n{'='*60}")
    print("SPEARMAN CORRELATIONS WITH AUC GAP")
    print(f"{'='*60}")
    print(f"  {'Predictor':25s}  {'rho':>7s}  {'p':>8s}  {'sig':>4s}")
    print(f"  {'-'*50}")

    predictors = [
        ("dt_score", "Directional transport"),
        ("dt_abs", "Dir. transport (abs)"),
        ("dt_alignment", "Discriminant alignment"),
        ("bn_score", "Bracket-norm"),
        ("geodesic", "Geodesic (symmetric)"),
        ("centroid_dist", "Centroid distance"),
    ]
    for col, label in predictors:
        rho, p = spearmanr(df[col], df["auc_gap"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:25s}  {rho:+.3f}  {p:8.4f}  {sig:>4s}")

    print(f"\n  n = {len(df)} ordered pairs")


def run(k=K, min_prevalence=0.1):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cohorts = load_crc_cohorts(min_prevalence=min_prevalence)
    df = run_ordered_pairs(cohorts, k=k)

    df.to_csv(RESULTS_DIR / "ordered_pairs.csv", index=False)
    print(f"\n  Saved ordered_pairs.csv ({len(df)} pairs)")

    print_correlations(df)

    summary = {
        "n_studies": len(set(df["from"])),
        "n_pairs": len(df),
        "k": k,
        "min_prevalence": min_prevalence,
    }
    predictors = ["dt_score", "dt_abs", "dt_alignment", "bn_score", "geodesic", "centroid_dist"]
    for col in predictors:
        rho, p = spearmanr(df[col], df["auc_gap"])
        summary[f"rho_{col}"] = float(rho)
        summary[f"p_{col}"] = float(p)

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{datetime.now():%H:%M:%S}] Done. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--min-prevalence", type=float, default=0.1)
    args = parser.parse_args()
    run(k=args.k, min_prevalence=args.min_prevalence)

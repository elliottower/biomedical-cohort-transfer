"""Additional robustness analyses for paper_v2c.

1. Clustered bootstrap (resample studies, not pairs) for CRC and QMDiab
2. Chordal distance comparison for CRC
3. RF classifier on CRC

Usage: PYTHONPATH=. uv run python robustness_extras.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, rankdata
from scipy.stats import t as t_dist
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.transportability import top_k_subspace, geodesic_distance, centroid_distance
from src.external_validation import external_validation, internal_validation
import importlib.util
_spec = importlib.util.spec_from_file_location("run_crc", "microbiome-crc/run_crc.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_crc_cohorts = _mod.load_crc_cohorts

RESULTS_DIR = Path("results/robustness_extras")
K = 10
SEED = 42


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-12:
        return np.nan
    return (r_xy - r_xz * r_yz) / denom


def clustered_bootstrap(df, study_col="from", n_boot=2000, seed=42):
    """Bootstrap resampling at the study level, not the pair level.

    For each bootstrap iteration:
    1. Resample the unique study IDs with replacement
    2. Include all pairs where BOTH source and target are in the resampled set
    3. Handle duplicates: if study A is sampled twice, pairs involving A appear twice
    4. Compute partial Spearman rho on the resampled pairs
    """
    print(f"[{datetime.now():%H:%M:%S}] Running clustered bootstrap ({n_boot} samples)...")
    rng = np.random.default_rng(seed)

    studies = sorted(df[study_col].unique())
    n_studies = len(studies)

    # Pre-index: for each (source, target) pair, store the row indices
    pair_idx = {}
    for idx, row in df.iterrows():
        key = (row["from"], row["to"])
        pair_idx.setdefault(key, []).append(idx)

    partials = []
    for _ in tqdm(range(n_boot), desc="Clustered bootstrap"):
        # Resample studies with replacement
        boot_studies = rng.choice(studies, size=n_studies, replace=True)

        # Collect all pairs where both from and to are in boot_studies
        # Handle duplicates: if study A appears k times, pairs with A as source
        # or target get counted k_from * k_to times
        from collections import Counter
        study_counts = Counter(boot_studies)

        boot_indices = []
        for (s_from, s_to), idxs in pair_idx.items():
            if s_from in study_counts and s_to in study_counts:
                n_reps = study_counts[s_from] * study_counts[s_to]
                boot_indices.extend(idxs * n_reps)

        if len(boot_indices) < 10:
            continue

        boot_df = df.iloc[boot_indices]
        rp = partial_spearman(
            boot_df["geodesic"].values,
            boot_df["auc_gap"].values,
            boot_df["int_auc"].values,
        )
        if not np.isnan(rp):
            partials.append(rp)

    partials = np.array(partials)
    ci_lo, ci_hi = np.percentile(partials, [2.5, 97.5])
    median = np.median(partials)
    pct_positive = (partials > 0).mean() * 100

    print(f"  Median partial rho: {median:+.3f}")
    print(f"  95% CI (clustered): [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"  {pct_positive:.1f}% positive")
    print(f"  {len(partials)}/{n_boot} valid samples")

    return {
        "median": float(median),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "pct_positive": float(pct_positive),
        "n_boot": n_boot,
        "n_valid": len(partials),
        "n_studies": n_studies,
    }


def chordal_distance(U1, U2):
    """Chordal distance: ||U1 U1^T - U2 U2^T||_F."""
    P1 = U1 @ U1.T
    P2 = U2 @ U2.T
    return np.linalg.norm(P1 - P2, "fro")


def run_crc_extras():
    """Chordal distance + RF classifier + clustered bootstrap for CRC."""
    print(f"\n{'='*60}")
    print("CRC EXTRAS")
    print(f"{'='*60}")

    cohorts = load_crc_cohorts(min_prevalence=0.1)
    studies = sorted(cohorts.keys())
    n = len(studies)

    # Compute chordal distance and RF AUC gap for all ordered pairs
    print(f"\n[{datetime.now():%H:%M:%S}] Computing chordal distance + RF for {n*(n-1)} pairs...")
    rows = []
    for i in tqdm(range(n), desc="CRC pairs"):
        for j in range(n):
            if i == j:
                continue
            s1, s2 = studies[i], studies[j]
            X1, y1 = cohorts[s1]["X"], cohorts[s1]["y"]
            X2, y2 = cohorts[s2]["X"], cohorts[s2]["y"]

            U1, _ = top_k_subspace(X1, K)
            U2, _ = top_k_subspace(X2, K)
            gdist = geodesic_distance(U1, U2)
            cdist_chordal = chordal_distance(U1, U2)
            cdist_centroid = centroid_distance(X1, X2)

            # LR
            ext_lr = external_validation(X1, y1, X2, y2, n_bootstrap=0, seed=SEED)
            int_lr = internal_validation(X1, y1, seed=SEED)
            gap_lr = int_lr["mean_auc"] - ext_lr["forward"]["auc"]

            # RF
            rf = RandomForestClassifier(n_estimators=100, random_state=SEED, max_depth=5)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            int_aucs_rf = []
            for train_idx, val_idx in skf.split(X1, y1):
                rf_fold = RandomForestClassifier(n_estimators=100, random_state=SEED, max_depth=5)
                rf_fold.fit(X1[train_idx], y1[train_idx])
                pred = rf_fold.predict_proba(X1[val_idx])[:, 1]
                int_aucs_rf.append(roc_auc_score(y1[val_idx], pred))
            int_auc_rf = np.mean(int_aucs_rf)

            rf.fit(X1, y1)
            ext_pred_rf = rf.predict_proba(X2)[:, 1]
            ext_auc_rf = roc_auc_score(y2, ext_pred_rf)
            gap_rf = int_auc_rf - ext_auc_rf

            rows.append({
                "from": s1, "to": s2,
                "geodesic": gdist,
                "chordal": cdist_chordal,
                "centroid_dist": cdist_centroid,
                "auc_gap": gap_lr,
                "int_auc": int_lr["mean_auc"],
                "ext_auc": ext_lr["forward"]["auc"],
                "auc_gap_rf": gap_rf,
                "int_auc_rf": int_auc_rf,
                "ext_auc_rf": ext_auc_rf,
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "crc_extras_ordered_pairs.csv", index=False)

    # Print correlations
    print(f"\n--- LR correlations ---")
    for col, label in [("geodesic", "Geodesic"), ("chordal", "Chordal"), ("centroid_dist", "Centroid")]:
        rho, p = spearmanr(df[col], df["auc_gap"])
        rp = partial_spearman(df[col].values, df["auc_gap"].values, df["int_auc"].values)
        print(f"  {label:12s}  raw={rho:+.3f} (p={p:.4f})  partial={rp:+.3f}")

    print(f"\n--- RF correlations ---")
    for col, label in [("geodesic", "Geodesic"), ("chordal", "Chordal"), ("centroid_dist", "Centroid")]:
        rho, p = spearmanr(df[col], df["auc_gap_rf"])
        rp = partial_spearman(df[col].values, df["auc_gap_rf"].values, df["int_auc_rf"].values)
        print(f"  {label:12s}  raw={rho:+.3f} (p={p:.4f})  partial={rp:+.3f}")

    # Clustered bootstrap - LR
    print(f"\n--- Clustered bootstrap (LR) ---")
    boot_lr = clustered_bootstrap(df, n_boot=2000, seed=SEED)

    # Clustered bootstrap - RF
    print(f"\n--- Clustered bootstrap (RF) ---")
    df_rf = df.rename(columns={"auc_gap_rf": "auc_gap_orig", "auc_gap": "auc_gap_lr"})
    df_rf["auc_gap"] = df["auc_gap_rf"]
    df_rf["int_auc"] = df["int_auc_rf"]
    boot_rf = clustered_bootstrap(df_rf, n_boot=2000, seed=SEED)

    # Chordal clustered bootstrap
    print(f"\n--- Clustered bootstrap (Chordal + LR) ---")
    df_chordal = df.copy()
    df_chordal["geodesic"] = df_chordal["chordal"]
    boot_chordal = clustered_bootstrap(df_chordal, n_boot=2000, seed=SEED)

    results = {
        "lr_geodesic": {
            "raw_rho": float(spearmanr(df["geodesic"], df["auc_gap"])[0]),
            "partial_rho": float(partial_spearman(df["geodesic"].values, df["auc_gap"].values, df["int_auc"].values)),
            "clustered_bootstrap": boot_lr,
        },
        "lr_chordal": {
            "raw_rho": float(spearmanr(df["chordal"], df["auc_gap"])[0]),
            "partial_rho": float(partial_spearman(df["chordal"].values, df["auc_gap"].values, df["int_auc"].values)),
            "clustered_bootstrap": boot_chordal,
        },
        "rf_geodesic": {
            "raw_rho": float(spearmanr(df["geodesic"], df["auc_gap_rf"])[0]),
            "partial_rho": float(partial_spearman(df["geodesic"].values, df["auc_gap_rf"].values, df["int_auc_rf"].values)),
            "clustered_bootstrap": boot_rf,
        },
    }

    with open(RESULTS_DIR / "crc_extras.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved crc_extras.json")
    return results


def run_qmdiab_clustered():
    """Clustered bootstrap for QMDiab."""
    print(f"\n{'='*60}")
    print("QMDIAB CLUSTERED BOOTSTRAP")
    print(f"{'='*60}")

    df = pd.read_csv("metabolomics-multilab/results/qmdiab_cross_platform_ordered_pairs.csv")
    boot = clustered_bootstrap(df, n_boot=2000, seed=SEED)

    with open(RESULTS_DIR / "qmdiab_clustered_bootstrap.json", "w") as f:
        json.dump(boot, f, indent=2)
    print(f"  Saved qmdiab_clustered_bootstrap.json")
    return boot


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    crc = run_crc_extras()
    qmdiab = run_qmdiab_clustered()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"CRC geodesic (LR):  partial rho = {crc['lr_geodesic']['partial_rho']:+.3f}, "
          f"clustered 95% CI [{crc['lr_geodesic']['clustered_bootstrap']['ci_lo']:+.3f}, "
          f"{crc['lr_geodesic']['clustered_bootstrap']['ci_hi']:+.3f}]")
    print(f"CRC chordal (LR):   partial rho = {crc['lr_chordal']['partial_rho']:+.3f}, "
          f"clustered 95% CI [{crc['lr_chordal']['clustered_bootstrap']['ci_lo']:+.3f}, "
          f"{crc['lr_chordal']['clustered_bootstrap']['ci_hi']:+.3f}]")
    print(f"CRC geodesic (RF):  partial rho = {crc['rf_geodesic']['partial_rho']:+.3f}, "
          f"clustered 95% CI [{crc['rf_geodesic']['clustered_bootstrap']['ci_lo']:+.3f}, "
          f"{crc['rf_geodesic']['clustered_bootstrap']['ci_hi']:+.3f}]")
    print(f"QMDiab geodesic:    clustered 95% CI [{qmdiab['ci_lo']:+.3f}, {qmdiab['ci_hi']:+.3f}]")
    print(f"\n[{datetime.now():%H:%M:%S}] All done.")

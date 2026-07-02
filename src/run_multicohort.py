"""Multi-cohort geometric transportability analysis.

Uses all 15 plates from MTBLS7260 as individual cohorts, testing pairwise
and cyclic sheaf consistency. This is the full Grassmannian analysis —
the 2-cohort split is a simplification.

Usage: PYTHONPATH=. python src/run_multicohort.py [--n-perm 100]
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.transportability import (
    sheaf_h1_multi_cohort, sheaf_h1_two_cohort,
    top_k_subspace, geodesic_distance, principal_angles,
    fisher_rao_confound_score,
)
from src.external_validation import external_validation, internal_validation
from src.nulls import null_cohort_labels
from src.embed import embed
from src.preprocess import log_transform
from src import figures


RESULTS_DIR = Path("results/multicohort")
K = 10
SEED = 42


def load_plate_cohorts(data_dir="data"):
    """Load all plates as separate cohorts from processed CSVs."""
    print(f"[{datetime.now():%H:%M:%S}] Loading plate-level cohorts...")

    df1 = pd.read_csv(Path(data_dir) / "processed/cohort1.csv")
    df2 = pd.read_csv(Path(data_dir) / "processed/cohort2.csv")
    df_all = pd.concat([df1, df2], ignore_index=True)

    meta_cols = {"label", "sample_id", "plate"}
    feat_cols = [c for c in df_all.columns if c not in meta_cols]

    plates = sorted(df_all["plate"].unique())
    cohorts = {}
    for plate in plates:
        mask = df_all["plate"] == plate
        X = df_all.loc[mask, feat_cols].values.astype(np.float64)
        y = df_all.loc[mask, "label"].values.astype(int)

        if np.any(X < 0):
            pass
        else:
            X = np.log2(X + 1.0)

        cohorts[plate] = {"X": X, "y": y, "n": len(y), "n_pos": int(y.sum())}
        print(f"  {plate}: {len(y)} samples ({y.sum():.0f} cancer)")

    return cohorts, feat_cols


def pairwise_geometry(cohorts, backend="raw_features"):
    """Compute pairwise geodesic distances and AUC gaps."""
    print(f"\n[{datetime.now():%H:%M:%S}] Pairwise geometry + AUC gaps...")

    plates = sorted(cohorts.keys())
    n = len(plates)

    subspaces = {}
    for plate in tqdm(plates, desc="  PCA subspaces"):
        X_emb = embed(cohorts[plate]["X"], backend=backend)
        U, _ = top_k_subspace(X_emb, K)
        subspaces[plate] = (U, X_emb)

    rows = []
    for i in tqdm(range(n), desc="  Pairwise"):
        for j in range(i + 1, n):
            p1, p2 = plates[i], plates[j]
            U1, X1 = subspaces[p1]
            U2, X2 = subspaces[p2]
            y1 = cohorts[p1]["y"]
            y2 = cohorts[p2]["y"]

            gdist = geodesic_distance(U1, U2)
            angles = principal_angles(U1, U2)

            ext = external_validation(X1, y1, X2, y2, n_bootstrap=0, seed=SEED)
            auc_int = internal_validation(X1, y1, seed=SEED)

            rows.append({
                "plate_1": p1,
                "plate_2": p2,
                "n_1": len(y1),
                "n_2": len(y2),
                "geodesic_dist": gdist,
                "mean_angle_deg": float(np.degrees(angles).mean()),
                "max_angle_deg": float(np.degrees(angles).max()),
                "min_angle_deg": float(np.degrees(angles).min()),
                "external_auc_fwd": ext["forward"]["auc"],
                "external_auc_rev": ext["reverse"]["auc"],
                "internal_auc": auc_int["mean_auc"],
                "auc_gap": auc_int["mean_auc"] - ext["forward"]["auc"],
            })

    return pd.DataFrame(rows), subspaces


def cyclic_holonomy(cohorts, subspaces):
    """Compute holonomy around cycles of plates."""
    print(f"\n[{datetime.now():%H:%M:%S}] Cyclic holonomy...")

    plates = sorted(cohorts.keys())
    Us = [subspaces[p][0] for p in plates]

    full = sheaf_h1_multi_cohort(
        [cohorts[p]["X"] for p in plates], k=K
    )
    print(f"  Full cycle ({len(plates)} plates): holonomy = {full['holonomy_norm']:.4f}")

    subcycles = []
    for size in [3, 5]:
        for start in range(0, len(plates) - size + 1, size):
            cycle_plates = plates[start:start + size]
            cycle_Us = [subspaces[p][0] for p in cycle_plates]
            sub = sheaf_h1_multi_cohort(
                [cohorts[p]["X"] for p in cycle_plates], k=K
            )
            subcycles.append({
                "cycle": " -> ".join(cycle_plates),
                "size": size,
                "holonomy_norm": sub["holonomy_norm"],
            })
            print(f"  Subcycle {cycle_plates}: holonomy = {sub['holonomy_norm']:.4f}")

    return full, pd.DataFrame(subcycles)


def null_model(cohorts, n_perm=100):
    """Permutation null for multi-cohort geodesic distances."""
    print(f"\n[{datetime.now():%H:%M:%S}] Multi-cohort null model ({n_perm} perms)...")

    plates = sorted(cohorts.keys())
    X_all = np.vstack([cohorts[p]["X"] for p in plates])
    labels = np.concatenate([np.full(cohorts[p]["n"], i) for i, p in enumerate(plates)])

    observed_subspaces = []
    for p in plates:
        mask = labels == plates.index(p)
        X_plate = embed(X_all[mask], backend="raw_features")
        U, _ = top_k_subspace(X_plate, K)
        observed_subspaces.append(U)

    observed_dists = []
    for i in range(len(plates)):
        for j in range(i + 1, len(plates)):
            observed_dists.append(geodesic_distance(
                observed_subspaces[i], observed_subspaces[j]
            ))
    observed_mean = np.mean(observed_dists)

    rng = np.random.default_rng(SEED)
    null_means = []
    for _ in tqdm(range(n_perm), desc="  null: plate labels"):
        perm = rng.permutation(labels)
        perm_subspaces = []
        for i in range(len(plates)):
            mask = perm == i
            X_group = embed(X_all[mask], backend="raw_features")
            U, _ = top_k_subspace(X_group, K)
            perm_subspaces.append(U)

        dists = []
        for i in range(len(plates)):
            for j in range(i + 1, len(plates)):
                dists.append(geodesic_distance(
                    perm_subspaces[i], perm_subspaces[j]
                ))
        null_means.append(np.mean(dists))

    null_means = np.array(null_means)
    z = (observed_mean - null_means.mean()) / (null_means.std() + 1e-12)
    p = float(np.mean(null_means >= observed_mean))

    print(f"  Observed mean geodesic: {observed_mean:.3f}")
    print(f"  Null mean: {null_means.mean():.3f} +/- {null_means.std():.3f}")
    print(f"  z = {z:.2f}, p = {p:.4f}")

    return {
        "observed_mean_geodesic": float(observed_mean),
        "null_mean": float(null_means.mean()),
        "null_std": float(null_means.std()),
        "z": float(z),
        "p": float(p),
        "null_distribution": null_means.tolist(),
    }


def run(n_perm=100):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)

    cohorts, feat_cols = load_plate_cohorts()

    pairwise_df, subspaces = pairwise_geometry(cohorts)
    pairwise_df.to_csv(RESULTS_DIR / "pairwise_geometry.csv", index=False)
    print(f"\n  Saved pairwise_geometry.csv ({len(pairwise_df)} pairs)")

    full_hol, subcycles_df = cyclic_holonomy(cohorts, subspaces)
    subcycles_df.to_csv(RESULTS_DIR / "subcycles.csv", index=False)

    null_results = null_model(cohorts, n_perm=n_perm)
    with open(RESULTS_DIR / "null_model.json", "w") as f:
        json.dump(null_results, f, indent=2)

    # F1: Geodesic distance vs AUC gap (money plot)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pairwise_df["geodesic_dist"], pairwise_df["auc_gap"],
               alpha=0.5, s=30)
    ax.set_xlabel("Grassmannian geodesic distance")
    ax.set_ylabel("AUC gap (internal - external)")
    ax.set_title("Geometric distance predicts generalization gap")
    r = np.corrcoef(pairwise_df["geodesic_dist"], pairwise_df["auc_gap"])[0, 1]
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
            va="top", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures/F1_geodesic_vs_gap.png", dpi=150)
    plt.close(fig)
    print(f"\n  Correlation (geodesic vs AUC gap): r = {r:.3f}")

    # F4: Pairwise distance heatmap
    plates = sorted(cohorts.keys())
    n = len(plates)
    dist_matrix = np.zeros((n, n))
    for _, row in pairwise_df.iterrows():
        i = plates.index(row["plate_1"])
        j = plates.index(row["plate_2"])
        dist_matrix[i, j] = row["geodesic_dist"]
        dist_matrix[j, i] = row["geodesic_dist"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dist_matrix, cmap="YlOrRd")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(plates, rotation=45, ha="right")
    ax.set_yticklabels(plates)
    ax.set_title("Pairwise Grassmannian geodesic distance")
    fig.colorbar(im, label="Geodesic distance")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures/F4_distance_heatmap.png", dpi=150)
    plt.close(fig)

    # Null distribution figure
    figures.plot_null_distributions(
        null_results["observed_mean_geodesic"],
        null_results["null_distribution"],
        "Mean pairwise geodesic distance",
        str(RESULTS_DIR / "figures/F3_null_multicohort.png"),
    )

    # Summary
    print(f"\n{'='*60}")
    print("MULTI-COHORT RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {len(cohorts)} plates, {sum(c['n'] for c in cohorts.values())} total samples")
    print(f"  {len(pairwise_df)} pairwise comparisons")
    print(f"  Mean geodesic distance: {pairwise_df['geodesic_dist'].mean():.3f}")
    print(f"  Mean AUC gap: {pairwise_df['auc_gap'].mean():.3f}")
    print(f"  Correlation (geodesic vs gap): r = {r:.3f}")
    print(f"  Full-cycle holonomy: {full_hol['holonomy_norm']:.4f}")
    print(f"  Null model: z = {null_results['z']:.2f}, p = {null_results['p']:.4f}")
    print(f"{'='*60}")
    print(f"\n[{datetime.now():%H:%M:%S}] Done. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=100)
    args = parser.parse_args()
    run(n_perm=args.n_perm)

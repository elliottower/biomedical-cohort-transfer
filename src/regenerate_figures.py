"""Regenerate all figures from saved results.

Usage: PYTHONPATH=. uv run python src/regenerate_figures.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src import figures
from src.transportability import sheaf_h1_two_cohort, top_k_subspace, geodesic_distance
from src.preprocess import load_feature_matrix, preprocess_cohorts
from src.embed import embed


RESULTS = Path("results")
MULTI = RESULTS / "multicohort"
K = 10


def regenerate_two_cohort():
    print(f"[{datetime.now():%H:%M:%S}] Regenerating 2-cohort figures...")
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df1 = load_feature_matrix("data/processed/cohort1.csv")
    df2 = load_feature_matrix("data/processed/cohort2.csv")
    data = preprocess_cohorts(df1, df2)

    X1_raw = embed(data["X1"], backend="raw_features")
    X2_raw = embed(data["X2"], backend="raw_features")
    X1_pca = embed(data["X1"], backend="classical", method="pca", n_components=50)
    X2_pca = embed(data["X2"], backend="classical", method="pca", n_components=50)

    geom_raw = sheaf_h1_two_cohort(X1_raw, X2_raw, k=K)
    geom_pca = sheaf_h1_two_cohort(X1_pca, X2_pca, k=K)

    # F2: Principal angle spectrum — raw vs PCA
    figures.plot_principal_angles(
        [geom_raw["principal_angles"], geom_pca["principal_angles"]],
        [f"Raw features (d={geom_raw['geodesic_dist']:.2f})",
         f"PCA-50 (d={geom_pca['geodesic_dist']:.2f})"],
        fig_dir / "F2_principal_angles.png"
    )

    # F2b: Explained variance comparison
    figures.plot_explained_variance(
        [geom_raw["explained_var_c1"], geom_raw["explained_var_c2"]],
        ["Cohort 1", "Cohort 2"],
        fig_dir / "F2b_explained_variance.png"
    )

    # F3: Null distributions (load from saved tables if available)
    null_file = RESULTS / "null_geodesic.json"
    if null_file.exists():
        null = json.loads(null_file.read_text())
        figures.plot_null_distributions(
            null["observed_geodesic"], null["null_geodesic"],
            "Geodesic distance", fig_dir / "F3_null_geodesic.png"
        )


def regenerate_multicohort():
    print(f"\n[{datetime.now():%H:%M:%S}] Regenerating multicohort figures...")
    fig_dir = MULTI / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pairwise = pd.read_csv(MULTI / "pairwise_geometry.csv")
    subcycles = pd.read_csv(MULTI / "subcycles.csv")
    null_path = MULTI / "null_model.json"

    # F1: Money plot — geodesic vs AUC gap
    figures.plot_geom_vs_gap(
        pairwise["geodesic_dist"].values,
        pairwise["auc_gap"].values,
        [f"{r.plate_1}-{r.plate_2}" for _, r in pairwise.iterrows()],
        fig_dir / "F1_geodesic_vs_gap.png",
        title="Subspace distance vs cross-cohort AUC gap (105 plate pairs)"
    )

    # F4: Pairwise distance heatmap with annotations
    plates = sorted(set(pairwise["plate_1"]) | set(pairwise["plate_2"]))
    n = len(plates)
    dist_matrix = np.zeros((n, n))
    for _, row in pairwise.iterrows():
        i = plates.index(row["plate_1"])
        j = plates.index(row["plate_2"])
        dist_matrix[i, j] = row["geodesic_dist"]
        dist_matrix[j, i] = row["geodesic_dist"]

    figures.plot_distance_heatmap(
        dist_matrix, plates, fig_dir / "F4_distance_heatmap.png"
    )

    # F5: AUC gap heatmap (z-score style, but using raw AUC gaps)
    gap_matrix = np.full((n, n), np.nan)
    for _, row in pairwise.iterrows():
        i = plates.index(row["plate_1"])
        j = plates.index(row["plate_2"])
        gap_matrix[i, j] = row["auc_gap"]
        gap_matrix[j, i] = -row["auc_gap"]  # reverse direction

    figures.plot_zscore_heatmap(
        gap_matrix, plates, plates, fig_dir / "F5_auc_gap_heatmap.png",
        title="Pairwise AUC gap (train row $\\rightarrow$ test column)",
        cbar_label="AUC gap"
    )

    # F6: Holonomy bar chart
    figures.plot_holonomy_bar(subcycles, fig_dir / "F6_holonomy_subcycles.png")

    # F3: Null distribution (multicohort) — prefer Modal 1000-perm data
    modal_null = RESULTS / "null_model_modal.json"
    best_null = modal_null if modal_null.exists() else null_path
    if best_null.exists():
        null = json.loads(best_null.read_text())
        figures.plot_null_distributions(
            null["observed_mean_geodesic"], null["null_distribution"],
            "Mean pairwise geodesic distance",
            fig_dir / "F3_null_multicohort.png"
        )

    # Multi-panel null grid: load per-plate null data if available
    # For now, just regenerate the main null
    print(f"  Generated {len(list(fig_dir.glob('*.png')))} figures")


def regenerate_combined():
    """Combined figures spanning both analyses."""
    print(f"\n[{datetime.now():%H:%M:%S}] Regenerating combined figures...")
    fig_dir = RESULTS / "figures"

    # Load multicohort pairwise data
    pairwise = pd.read_csv(MULTI / "pairwise_geometry.csv")

    # Principal angle spectra for selected plate pairs (most/least divergent)
    df1 = load_feature_matrix("data/processed/cohort1.csv")
    df2 = load_feature_matrix("data/processed/cohort2.csv")
    df_all = pd.concat([df1, df2], ignore_index=True)

    meta_cols = {"label", "sample_id", "plate"}
    feat_cols = [c for c in df_all.columns if c not in meta_cols]

    best_pair = pairwise.loc[pairwise["geodesic_dist"].idxmin()]
    worst_pair = pairwise.loc[pairwise["geodesic_dist"].idxmax()]

    angles_list = []
    labels_list = []
    for pair, desc in [(best_pair, "closest"), (worst_pair, "farthest")]:
        p1, p2 = pair["plate_1"], pair["plate_2"]
        X1 = df_all.loc[df_all["plate"] == p1, feat_cols].values.astype(np.float64)
        X2 = df_all.loc[df_all["plate"] == p2, feat_cols].values.astype(np.float64)
        X1 = np.log2(X1 + 1.0)
        X2 = np.log2(X2 + 1.0)
        X1_emb = embed(X1, backend="raw_features")
        X2_emb = embed(X2, backend="raw_features")
        geom = sheaf_h1_two_cohort(X1_emb, X2_emb, k=K)
        angles_list.append(geom["principal_angles"])
        labels_list.append(f"{p1} vs {p2} ({desc}, d={geom['geodesic_dist']:.2f})")

    figures.plot_principal_angles(
        angles_list, labels_list,
        fig_dir / "F2c_angle_spectrum_extremes.png"
    )


if __name__ == "__main__":
    regenerate_two_cohort()
    regenerate_multicohort()
    regenerate_combined()
    print(f"\n[{datetime.now():%H:%M:%S}] Done.")

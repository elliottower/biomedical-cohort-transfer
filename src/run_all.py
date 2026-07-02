"""End-to-end orchestrator: download -> preprocess -> embed -> validate -> geometry -> nulls -> figures.

Usage: python src/run_all.py [--config config.yaml] [--backend raw_features]
"""
import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.preprocess import load_feature_matrix, preprocess_cohorts
from src.embed import embed
from src.transportability import sheaf_h1_two_cohort, fisher_rao_confound_score
from src.external_validation import external_validation, internal_validation
from src.nulls import null_cohort_labels, null_clinical_labels
from src import figures


def save_csv(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict):
        pd.DataFrame([data]).to_csv(path, index=False)
    elif isinstance(data, list):
        pd.DataFrame(data).to_csv(path, index=False)
    else:
        data.to_csv(path, index=False)
    print(f"  Saved {path}")


def run(config_path="config.yaml", backend="raw_features"):
    print(f"[{datetime.now():%H:%M:%S}] Starting geometric transportability pipeline")
    print(f"  Backend: {backend}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path("results")
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    k = cfg["transportability"]["subspace_k"]
    n_perm = cfg["transportability"]["n_permutations"]
    classifier = cfg["external_validation"]["classifier"]
    n_boot = cfg["external_validation"]["n_bootstrap"]

    # --- Load data ---
    print(f"\n[{datetime.now():%H:%M:%S}] Loading data...")
    data_dir = Path(cfg["dataset"]["data_dir"]) / "processed"
    c1_path = data_dir / "cohort1.csv"
    c2_path = data_dir / "cohort2.csv"

    if not c1_path.exists() or not c2_path.exists():
        print(f"  ERROR: Processed data not found at {data_dir}/")
        print("  Run preprocessing first or place cohort1.csv and cohort2.csv in data/processed/")
        return

    df1 = load_feature_matrix(c1_path)
    df2 = load_feature_matrix(c2_path)
    data = preprocess_cohorts(df1, df2)

    # --- T1: Cohort summary ---
    t1 = [
        {"cohort": "C1", "n": data["X1"].shape[0], "n_features": data["n_features"],
         "pct_positive": float(data["y1"].mean()) * 100},
        {"cohort": "C2", "n": data["X2"].shape[0], "n_features": data["n_features"],
         "pct_positive": float(data["y2"].mean()) * 100},
    ]
    save_csv(t1, tables_dir / "T1_cohort_summary.csv")

    # --- Embed ---
    print(f"\n[{datetime.now():%H:%M:%S}] Embedding ({backend})...")
    embed_kwargs = {}
    if backend == "classical":
        embed_kwargs = cfg["embedding"].get("classical", {})

    X1_emb = embed(data["X1"], backend=backend, **embed_kwargs)
    X2_emb = embed(data["X2"], backend=backend, **embed_kwargs)

    # --- T2: External validation ---
    print(f"\n[{datetime.now():%H:%M:%S}] External validation...")
    int_c1 = internal_validation(X1_emb, data["y1"], classifier=classifier, seed=seed)
    int_c2 = internal_validation(X2_emb, data["y2"], classifier=classifier, seed=seed)
    ext = external_validation(X1_emb, data["y1"], X2_emb, data["y2"],
                              classifier=classifier, n_bootstrap=n_boot, seed=seed)

    t2 = {
        "backend": backend,
        "internal_auc_c1": int_c1["mean_auc"],
        "internal_auc_c2": int_c2["mean_auc"],
        "external_auc_fwd": ext["forward"]["auc"],
        "external_auc_rev": ext["reverse"]["auc"],
        "external_auc_fwd_ci_lo": ext["forward"]["auc_ci_95"][0],
        "external_auc_fwd_ci_hi": ext["forward"]["auc_ci_95"][1],
        "auc_gap_fwd": int_c1["mean_auc"] - ext["forward"]["auc"],
        "auc_gap_rev": int_c2["mean_auc"] - ext["reverse"]["auc"],
    }
    save_csv(t2, tables_dir / "T2_external_validation.csv")

    # --- T3: Geometric scores ---
    print(f"\n[{datetime.now():%H:%M:%S}] Computing geometric transportability scores...")
    geom = sheaf_h1_two_cohort(X1_emb, X2_emb, k=k)

    fisher = None
    if data["batch1"] is not None and data["batch2"] is not None:
        fisher = fisher_rao_confound_score(
            X1_emb, X2_emb, data["batch1"], data["batch2"], data["y1"], data["y2"]
        )

    t3 = {
        "backend": backend,
        "geodesic_dist": geom["geodesic_dist"],
        "holonomy_norm": geom["holonomy_norm"],
        "h1_nonzero": geom["h1_nonzero"],
        "mean_principal_angle_deg": float(np.degrees(geom["principal_angles"]).mean()),
    }
    if fisher:
        t3["confound_fraction"] = fisher["confound_fraction"]
        t3["signal_fraction"] = fisher["signal_fraction"]
    save_csv(t3, tables_dir / "T3_geometric_scores.csv")

    # --- Null models ---
    print(f"\n[{datetime.now():%H:%M:%S}] Running null models ({n_perm} permutations)...")
    X_pooled = np.vstack([X1_emb, X2_emb])
    cohort_labels = np.concatenate([
        np.zeros(len(X1_emb)), np.ones(len(X2_emb))
    ]).astype(int)

    null_geom = null_cohort_labels(X_pooled, cohort_labels, k=k,
                                   n_permutations=n_perm, seed=seed)
    null_clin = null_clinical_labels(X1_emb, data["y1"], X2_emb, data["y2"],
                                     classifier=classifier, n_permutations=min(n_perm, 200),
                                     seed=seed)

    t3["geodesic_z"] = null_geom["geodesic_z"]
    t3["geodesic_p"] = null_geom["geodesic_p"]
    t3["holonomy_z"] = null_geom["holonomy_z"]
    t3["holonomy_p"] = null_geom["holonomy_p"]
    t3["auc_gap_z"] = null_clin["gap_z"]
    t3["auc_gap_p"] = null_clin["gap_p"]
    save_csv(t3, tables_dir / "T3_geometric_scores.csv")

    # --- T4: Prediction test ---
    t4 = {
        "backend": backend,
        "geodesic_dist": geom["geodesic_dist"],
        "auc_gap": t2["auc_gap_fwd"],
        "geodesic_z": null_geom["geodesic_z"],
        "geodesic_p": null_geom["geodesic_p"],
    }
    save_csv(t4, tables_dir / "T4_prediction_test.csv")

    # --- Figures ---
    print(f"\n[{datetime.now():%H:%M:%S}] Generating figures...")

    figures.plot_principal_angles(
        [geom["principal_angles"]],
        [f"C1 vs C2 ({backend})"],
        figures_dir / "F2_principal_angles.png"
    )

    figures.plot_null_distributions(
        null_geom["observed_geodesic"],
        null_geom["null_geodesic"],
        "Geodesic distance",
        figures_dir / "F3_null_geodesic.png"
    )

    figures.plot_null_distributions(
        null_geom["observed_holonomy"],
        null_geom["null_holonomy"],
        "Holonomy norm",
        figures_dir / "F3_null_holonomy.png"
    )

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY ({backend})")
    print(f"{'='*60}")
    print(f"  Internal AUC (C1): {int_c1['mean_auc']:.3f}")
    print(f"  Internal AUC (C2): {int_c2['mean_auc']:.3f}")
    print(f"  External AUC (C1->C2): {ext['forward']['auc']:.3f} "
          f"[{ext['forward']['auc_ci_95'][0]:.3f}, {ext['forward']['auc_ci_95'][1]:.3f}]")
    print(f"  External AUC (C2->C1): {ext['reverse']['auc']:.3f}")
    print(f"  AUC gap (fwd): {t2['auc_gap_fwd']:.3f}")
    print(f"  Geodesic distance: {geom['geodesic_dist']:.3f} (z={null_geom['geodesic_z']:.2f}, p={null_geom['geodesic_p']:.4f})")
    print(f"  Holonomy norm: {geom['holonomy_norm']:.3f} (z={null_geom['holonomy_z']:.2f}, p={null_geom['holonomy_p']:.4f})")
    print(f"  H^1 != 0: {geom['h1_nonzero']}")
    print(f"{'='*60}")

    print(f"\n[{datetime.now():%H:%M:%S}] Done. Results in {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backend", default="raw_features",
                        choices=["raw_features", "classical", "lsm"])
    args = parser.parse_args()
    run(config_path=args.config, backend=args.backend)

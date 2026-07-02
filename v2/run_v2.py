"""V2 multi-cohort analysis: adds dumb baselines, Fisher-Rao, Random Forest.

Extends the v1 pairwise geometry with:
  - 3 dumb baselines (centroid dist, domain classifier AUC, variance ratio)
  - Fisher-Rao confound score per pair
  - Random Forest external validation alongside logistic regression
  - T5 baseline comparison table
  - Updated F1 with baseline overlays

Usage: PYTHONPATH=. uv run python v2/run_v2.py [--n-perm 50]
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
    top_k_subspace, geodesic_distance, principal_angles,
    fisher_rao_confound_score, centroid_distance,
    domain_classifier_auc, variance_ratio,
)
from src.external_validation import external_validation, internal_validation
from src.embed import embed
from src.run_multicohort import load_plate_cohorts
from src import figures

RESULTS_DIR = Path("v2/results")
K = 10
SEED = 42


def pairwise_v2(cohorts):
    """Pairwise analysis with all geometric scores + baselines + Random Forest."""
    print(f"\n[{datetime.now():%H:%M:%S}] V2 pairwise analysis...")

    plates = sorted(cohorts.keys())
    n = len(plates)

    subspaces = {}
    for plate in tqdm(plates, desc="  PCA subspaces"):
        X_emb = embed(cohorts[plate]["X"], backend="raw_features")
        U, _ = top_k_subspace(X_emb, K)
        subspaces[plate] = U

    rows = []
    for i in tqdm(range(n), desc="  Pairwise v2"):
        for j in range(i + 1, n):
            p1, p2 = plates[i], plates[j]
            U1 = subspaces[p1]
            U2 = subspaces[p2]
            X1_raw = cohorts[p1]["X"]
            X2_raw = cohorts[p2]["X"]
            y1 = cohorts[p1]["y"]
            y2 = cohorts[p2]["y"]

            gdist = geodesic_distance(U1, U2)
            angles = principal_angles(U1, U2)

            ext_lr = external_validation(X1_raw, y1, X2_raw, y2,
                                         classifier="logistic",
                                         n_bootstrap=0, seed=SEED)
            ext_rf = external_validation(X1_raw, y1, X2_raw, y2,
                                          classifier="random_forest",
                                          n_bootstrap=0, seed=SEED)
            auc_int_lr = internal_validation(X1_raw, y1,
                                             classifier="logistic", seed=SEED)
            auc_int_rf = internal_validation(X1_raw, y1,
                                              classifier="random_forest", seed=SEED)

            batch_labels_1 = np.zeros(len(y1))
            batch_labels_2 = np.ones(len(y2))
            fr = fisher_rao_confound_score(X1_raw, X2_raw,
                                           batch_labels_1, batch_labels_2,
                                           y1, y2)

            c_dist = centroid_distance(X1_raw, X2_raw)
            d_auc = domain_classifier_auc(X1_raw, X2_raw)
            v_ratio = variance_ratio(X1_raw, X2_raw)

            rows.append({
                "plate_1": p1,
                "plate_2": p2,
                "n_1": len(y1),
                "n_2": len(y2),
                # Geometric scores
                "geodesic_dist": gdist,
                "mean_angle_deg": float(np.degrees(angles).mean()),
                "confound_fraction": fr["confound_fraction"],
                "signal_fraction": fr["signal_fraction"],
                "raw_shift": fr["raw_shift"],
                # Dumb baselines
                "centroid_dist": c_dist,
                "domain_auc": d_auc,
                "variance_ratio": v_ratio,
                # Logistic regression
                "ext_auc_lr_fwd": ext_lr["forward"]["auc"],
                "ext_auc_lr_rev": ext_lr["reverse"]["auc"],
                "int_auc_lr": auc_int_lr["mean_auc"],
                "auc_gap_lr": auc_int_lr["mean_auc"] - ext_lr["forward"]["auc"],
                # Random Forest
                "ext_auc_rf_fwd": ext_rf["forward"]["auc"],
                "ext_auc_rf_rev": ext_rf["reverse"]["auc"],
                "int_auc_rf": auc_int_rf["mean_auc"],
                "auc_gap_rf": auc_int_rf["mean_auc"] - ext_rf["forward"]["auc"],
            })

    return pd.DataFrame(rows), subspaces


def baseline_comparison(df):
    """T5: Compare geometric scores vs dumb baselines on prediction test."""
    print(f"\n[{datetime.now():%H:%M:%S}] Baseline comparison (T5)...")

    scores = {
        "geodesic_dist": "Grassmannian geodesic",
        "confound_fraction": "Fisher-Rao confound",
        "centroid_dist": "Centroid distance (baseline)",
        "domain_auc": "Domain classifier AUC (baseline)",
        "variance_ratio": "Variance ratio (baseline)",
    }

    rows = []
    for gap_col, clf_name in [("auc_gap_lr", "Logistic"), ("auc_gap_rf", "Random Forest")]:
        for col, label in scores.items():
            rho, p = spearmanr(df[col], df[gap_col])
            rows.append({
                "score": label,
                "classifier": clf_name,
                "spearman_rho": rho,
                "p_value": p,
                "abs_rho": abs(rho),
                "is_baseline": "baseline" in label.lower(),
            })
            print(f"  {label:40s} vs {clf_name:8s} gap: rho={rho:+.3f}, p={p:.3f}")

    t5 = pd.DataFrame(rows)

    geo_scores = t5[~t5["is_baseline"]]
    baselines = t5[t5["is_baseline"]]
    max_baseline_rho = baselines["abs_rho"].max()

    print(f"\n  Max baseline |rho|: {max_baseline_rho:.3f}")
    for _, row in geo_scores.iterrows():
        beats = row["abs_rho"] > max_baseline_rho
        status = "BEATS baselines" if beats else "reduces to baseline"
        print(f"  {row['score']:40s} ({row['classifier']}): |rho|={row['abs_rho']:.3f} -> {status}")

    return t5


def plot_scatter_with_baselines(df, outpath):
    """F1 v2: Geodesic vs gap scatter with baseline correlations overlaid."""
    import matplotlib.pyplot as plt
    figures.set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, gap_col, clf_name in [(axes[0], "auc_gap_lr", "Logistic Regression"),
                                   (axes[1], "auc_gap_rf", "Random Forest")]:
        gap = df[gap_col].values

        score_specs = [
            ("geodesic_dist", "Geodesic dist", "#2171b5", "o"),
            ("centroid_dist", "Centroid dist", "#cb181d", "s"),
            ("domain_auc", "Domain AUC", "#238b45", "^"),
        ]

        for col, label, color, marker in score_specs:
            x = df[col].values
            x_norm = (x - x.mean()) / (x.std() + 1e-12)
            rho, p = spearmanr(x, gap)
            ax.scatter(x_norm, gap, s=30, c=color, alpha=0.35,
                       marker=marker, edgecolors="none",
                       label=f"{label} ($\\rho={rho:+.2f}$)")

        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Score (standardized)")
        ax.set_ylabel("AUC gap (internal $-$ external)")
        ax.set_title(clf_name)
        ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("Geometric scores vs dumb baselines: predicting AUC gap", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


def run(n_perm=50):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)

    cohorts, feat_cols = load_plate_cohorts()

    pairwise_df, subspaces = pairwise_v2(cohorts)
    pairwise_df.to_csv(RESULTS_DIR / "tables" / "pairwise_v2.csv", index=False)
    print(f"\n  Saved pairwise_v2.csv ({len(pairwise_df)} pairs)")

    t5 = baseline_comparison(pairwise_df)
    t5.to_csv(RESULTS_DIR / "tables" / "T5_baseline_comparison.csv", index=False)
    print(f"  Saved T5_baseline_comparison.csv")

    plot_scatter_with_baselines(
        pairwise_df, RESULTS_DIR / "figures" / "F1_v2_scatter_with_baselines.png"
    )

    # Fisher-Rao summary
    print(f"\n  Fisher-Rao confound fraction: "
          f"mean={pairwise_df['confound_fraction'].mean():.3f}, "
          f"range=[{pairwise_df['confound_fraction'].min():.3f}, "
          f"{pairwise_df['confound_fraction'].max():.3f}]")

    # Random Forest vs LR comparison
    print(f"\n  LR  AUC gap: mean={pairwise_df['auc_gap_lr'].mean():.3f}")
    print(f"  RF AUC gap: mean={pairwise_df['auc_gap_rf'].mean():.3f}")

    rho_lr, p_lr = spearmanr(pairwise_df["geodesic_dist"], pairwise_df["auc_gap_lr"])
    rho_rf, p_rf = spearmanr(pairwise_df["geodesic_dist"], pairwise_df["auc_gap_rf"])
    print(f"\n  Geodesic vs LR gap:  rho={rho_lr:+.3f}, p={p_lr:.3f}")
    print(f"  Geodesic vs RF gap: rho={rho_rf:+.3f}, p={p_rf:.3f}")

    print(f"\n[{datetime.now():%H:%M:%S}] V2 analysis complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=50)
    args = parser.parse_args()
    run(n_perm=args.n_perm)

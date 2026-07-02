"""Leave-one-study-out prospective validation for CRC.

For each held-out study S:
  1. Train partial-correlation model on pairs not involving S
  2. Predict AUC gap for pairs involving S
  3. Report prediction error

Usage: PYTHONPATH=. uv run python batch2_extensions/loo_validation/run_loo.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from src.transportability import geodesic_distance, top_k_subspace, centroid_distance
from src.external_validation import external_validation, internal_validation
import importlib.util

_spec = importlib.util.spec_from_file_location("run_crc", "microbiome-crc/run_crc.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_crc_cohorts = _mod.load_crc_cohorts

RESULTS_DIR = Path("batch2_extensions/loo_validation/results")
FIG_DIR = Path("batch2_extensions/loo_validation/figures")
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


def compute_all_pairs(cohorts, k=K, seed=SEED):
    """Compute geodesic distance, AUC gap, internal AUC for all ordered pairs."""
    studies = sorted(cohorts.keys())
    rows = []
    for i in tqdm(range(len(studies)), desc="Computing pairs"):
        for j in range(len(studies)):
            if i == j:
                continue
            s1, s2 = studies[i], studies[j]
            X1, y1 = cohorts[s1]["X"], cohorts[s1]["y"]
            X2, y2 = cohorts[s2]["X"], cohorts[s2]["y"]

            U1, _ = top_k_subspace(X1, k)
            U2, _ = top_k_subspace(X2, k)
            gdist = geodesic_distance(U1, U2)
            cdist = centroid_distance(X1, X2)

            ext = external_validation(X1, y1, X2, y2, n_bootstrap=0, seed=seed)
            intv = internal_validation(X1, y1, seed=seed)
            gap = intv["mean_auc"] - ext["forward"]["auc"]

            rows.append({
                "from": s1, "to": s2,
                "geodesic": gdist,
                "centroid_dist": cdist,
                "auc_gap": gap,
                "int_auc": intv["mean_auc"],
                "ext_auc": ext["forward"]["auc"],
            })
    return pd.DataFrame(rows)


def run_loo_validation(df):
    """Leave-one-study-out: fit model on 8 studies, predict held-out study's pairs."""
    studies = sorted(set(df["from"].unique()) | set(df["to"].unique()))
    results = []

    for held_out in studies:
        train_mask = (df["from"] != held_out) & (df["to"] != held_out)
        test_mask = (df["from"] == held_out) | (df["to"] == held_out)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) < 10 or len(test_df) < 2:
            continue

        X_train = train_df[["int_auc", "geodesic"]].values
        y_train = train_df["auc_gap"].values

        X_test = test_df[["int_auc", "geodesic"]].values
        y_test = test_df["auc_gap"].values

        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        rho_pred, _ = spearmanr(y_test, y_pred)

        baseline_pred = train_df["auc_gap"].mean()
        mae_baseline = np.mean(np.abs(y_test - baseline_pred))

        int_only_reg = LinearRegression().fit(
            train_df[["int_auc"]].values, y_train
        )
        y_pred_int_only = int_only_reg.predict(test_df[["int_auc"]].values)
        mae_int_only = np.mean(np.abs(y_test - y_pred_int_only))

        results.append({
            "held_out": held_out,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "mae_full": float(mae),
            "rmse_full": float(rmse),
            "rho_predicted_vs_actual": float(rho_pred),
            "mae_baseline_mean": float(mae_baseline),
            "mae_int_only": float(mae_int_only),
            "train_partial_rho": float(partial_spearman(
                train_df["geodesic"].values,
                train_df["auc_gap"].values,
                train_df["int_auc"].values,
            )),
        })
        print(f"  {held_out}: MAE={mae:.3f} (baseline={mae_baseline:.3f}, "
              f"int-only={mae_int_only:.3f}), rho={rho_pred:.3f}")

    return results


def plot_loo_results(df, loo_results, outpath):
    """Scatter: predicted vs actual AUC gap across all LOO folds."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    studies = sorted(set(df["from"].unique()))
    all_actual, all_predicted, all_studies_label = [], [], []

    for held_out in studies:
        train_mask = (df["from"] != held_out) & (df["to"] != held_out)
        test_mask = (df["from"] == held_out) | (df["to"] == held_out)
        train_df = df[train_mask]
        test_df = df[test_mask]

        reg = LinearRegression().fit(
            train_df[["int_auc", "geodesic"]].values,
            train_df["auc_gap"].values,
        )
        y_pred = reg.predict(test_df[["int_auc", "geodesic"]].values)

        all_actual.extend(test_df["auc_gap"].values)
        all_predicted.extend(y_pred)
        all_studies_label.extend([held_out] * len(test_df))

    actual = np.array(all_actual)
    predicted = np.array(all_predicted)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(predicted, actual, s=25, alpha=0.5, c="steelblue", edgecolors="navy", linewidth=0.3)
    lims = [min(predicted.min(), actual.min()) - 0.05,
            max(predicted.max(), actual.max()) + 0.05]
    ax.plot(lims, lims, "--", color="gray", linewidth=1)
    ax.set_xlabel("Predicted AUC gap (LOO)")
    ax.set_ylabel("Actual AUC gap")
    rho, _ = spearmanr(actual, predicted)
    mae = np.mean(np.abs(actual - predicted))
    ax.set_title(f"LOO predicted vs actual: rho={rho:+.2f}, MAE={mae:.3f}")

    ax = axes[1]
    loo_df = pd.DataFrame(loo_results)
    x = range(len(loo_df))
    ax.bar(x, loo_df["mae_baseline_mean"], width=0.25, label="Baseline (mean)", color="#d9d9d9", align="center")
    ax.bar([i + 0.25 for i in x], loo_df["mae_int_only"], width=0.25, label="Int AUC only", color="#fdae6b", align="center")
    ax.bar([i + 0.5 for i in x], loo_df["mae_full"], width=0.25, label="Int AUC + geodesic", color="#31a354", align="center")
    ax.set_xticks([i + 0.25 for i in x])
    ax.set_xticklabels(loo_df["held_out"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE on held-out pairs")
    ax.set_title("LOO prediction error by held-out study")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] Loading CRC cohorts...")
    cohorts = load_crc_cohorts(min_prevalence=0.1)

    print(f"\n[{datetime.now():%H:%M:%S}] Computing all pairs...")
    df = compute_all_pairs(cohorts)
    df.to_csv(RESULTS_DIR / "loo_all_pairs.csv", index=False)

    print(f"\n[{datetime.now():%H:%M:%S}] Running LOO validation...")
    loo_results = run_loo_validation(df)

    with open(RESULTS_DIR / "loo_results.json", "w") as f:
        json.dump(loo_results, f, indent=2)

    summary = {
        "overall_mae_full": float(np.mean([r["mae_full"] for r in loo_results])),
        "overall_mae_int_only": float(np.mean([r["mae_int_only"] for r in loo_results])),
        "overall_mae_baseline": float(np.mean([r["mae_baseline_mean"] for r in loo_results])),
        "overall_rho": float(np.mean([r["rho_predicted_vs_actual"] for r in loo_results])),
        "mae_improvement_over_baseline_pct": float(
            (1 - np.mean([r["mae_full"] for r in loo_results]) /
             np.mean([r["mae_baseline_mean"] for r in loo_results])) * 100
        ),
        "mae_improvement_over_int_only_pct": float(
            (1 - np.mean([r["mae_full"] for r in loo_results]) /
             np.mean([r["mae_int_only"] for r in loo_results])) * 100
        ),
    }
    with open(RESULTS_DIR / "loo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- LOO Summary ---")
    print(f"  Mean MAE (full model):     {summary['overall_mae_full']:.4f}")
    print(f"  Mean MAE (int AUC only):   {summary['overall_mae_int_only']:.4f}")
    print(f"  Mean MAE (baseline mean):  {summary['overall_mae_baseline']:.4f}")
    print(f"  Improvement over baseline: {summary['mae_improvement_over_baseline_pct']:.1f}%")
    print(f"  Improvement over int-only: {summary['mae_improvement_over_int_only_pct']:.1f}%")
    print(f"  Mean LOO rho:              {summary['overall_rho']:.3f}")

    print(f"\n[{datetime.now():%H:%M:%S}] Plotting...")
    plot_loo_results(df, loo_results, FIG_DIR / "F1_loo_prospective.png")

    print(f"\n[{datetime.now():%H:%M:%S}] Done.")

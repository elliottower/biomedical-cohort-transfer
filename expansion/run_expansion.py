"""Expanded cross-cohort transportability analysis.

Runs the geodesic partial-correlation pipeline on expanded CRC and T2D
datasets from curatedMetagenomicData. Produces per-disease results
including partial correlations, clustered bootstrap CIs, k-sensitivity,
and LOO validation.

Usage:
    PYTHONPATH=. uv run python expansion/run_expansion.py --disease crc
    PYTHONPATH=. uv run python expansion/run_expansion.py --disease t2d
    PYTHONPATH=. uv run python expansion/run_expansion.py --disease all
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from src.transportability import geodesic_distance, top_k_subspace, centroid_distance
from src.external_validation import external_validation, internal_validation

SEED = 42
DATA_DIR = Path("expansion/data")
RESULTS_DIR = Path("expansion/results")
K_DEFAULT = 10
K_RANGE = [5, 10, 15, 20, 30, 50]
N_BOOTSTRAP = 2000
MIN_SAMPLES = 20
MIN_PREVALENCE = 0.1


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-12:
        return np.nan
    return (r_xy - r_xz * r_yz) / denom


def clustered_bootstrap_partial(df, n_boot=N_BOOTSTRAP, seed=SEED):
    """Clustered bootstrap resampling at the study level.

    Resamples study identities with replacement, then constructs the full
    ordered-pair set from the resampled list. A pair (A->B) where A is drawn
    m times and B is drawn n times appears m*n times in the bootstrap sample.
    """
    rng = np.random.default_rng(seed)
    studies = sorted(df["from_study"].unique())
    C = len(studies)
    partials = []
    for _ in range(n_boot):
        boot_studies = list(rng.choice(studies, size=C, replace=True))
        rows = []
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue
                mask = (df["from_study"] == boot_studies[i]) & (df["to_study"] == boot_studies[j])
                sub = df[mask]
                if len(sub) > 0:
                    rows.append(sub)
        if not rows:
            continue
        boot_df = pd.concat(rows, ignore_index=True)
        if len(boot_df) < 6:
            continue
        rho = partial_spearman(
            boot_df["geodesic"].values,
            boot_df["auc_gap"].values,
            boot_df["int_auc"].values,
        )
        if not np.isnan(rho):
            partials.append(rho)
    partials = np.array(partials)
    return {
        "median": float(np.median(partials)),
        "ci_lo": float(np.percentile(partials, 2.5)),
        "ci_hi": float(np.percentile(partials, 97.5)),
        "pct_positive": float(np.mean(partials > 0) * 100),
        "n_valid": len(partials),
    }


def load_disease(disease):
    """Load species profiles and metadata exported from R."""
    species_file = DATA_DIR / f"{disease}_species.tsv"
    meta_file = DATA_DIR / f"{disease}_metadata.tsv"

    if not species_file.exists():
        raise FileNotFoundError(
            f"{species_file} not found. Run export_cmd_data.R first."
        )

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Loading {disease} data...")

    species = pd.read_csv(species_file, sep="\t", index_col=0)
    meta = pd.read_csv(meta_file, sep="\t", index_col=0)

    common = species.columns.intersection(meta.index)
    species = species[common]
    meta = meta.loc[common]

    print(f"  {species.shape[0]} features x {species.shape[1]} samples")
    print(f"  Studies: {sorted(meta['study_name'].unique())}")

    prevalence = (species > 0).mean(axis=1)
    keep = prevalence >= MIN_PREVALENCE
    species = species.loc[keep]
    print(f"  {keep.sum()} features after {MIN_PREVALENCE:.0%} prevalence filter")

    X_all = species.values.T.astype(np.float64)
    X_all = np.log1p(X_all * 1e4)

    cohorts = {}
    for study in sorted(meta["study_name"].unique()):
        mask = (meta["study_name"] == study).values
        X = X_all[mask]
        y = meta.loc[mask, "is_case"].values.astype(int)
        if len(y) < MIN_SAMPLES or len(np.unique(y)) < 2:
            print(f"  Skipping {study}: {len(y)} samples, {np.unique(y)} classes")
            continue
        cohorts[study] = {"X": X, "y": y, "n": len(y)}
        n_case = int(y.sum())
        n_ctrl = int((1 - y).sum())
        print(f"  {study}: {len(y)} samples ({n_case} case, {n_ctrl} ctrl)")

    print(f"  {len(cohorts)} studies retained")
    return cohorts


def run_ordered_pairs(cohorts, k=K_DEFAULT):
    """Compute geodesic distances, AUC gaps, and partial correlations."""
    ts = datetime.now().strftime("%H:%M:%S")
    studies = sorted(cohorts.keys())
    n = len(studies)
    print(f"\n[{ts}] Running {n * (n - 1)} ordered pairs (k={k})...")

    rows = []
    for i in tqdm(range(n), desc="Ordered pairs"):
        for j in range(n):
            if i == j:
                continue
            s1, s2 = studies[i], studies[j]
            X1, y1 = cohorts[s1]["X"], cohorts[s1]["y"]
            X2, y2 = cohorts[s2]["X"], cohorts[s2]["y"]

            U1, _ = top_k_subspace(X1, k)
            U2, _ = top_k_subspace(X2, k)
            gdist = geodesic_distance(U1, U2)
            cdist = centroid_distance(X1, X2)

            ext = external_validation(X1, y1, X2, y2, n_bootstrap=0, seed=SEED)
            auc_int = internal_validation(X1, y1, seed=SEED)
            auc_gap = auc_int["mean_auc"] - ext["forward"]["auc"]

            rows.append({
                "from_study": s1,
                "to_study": s2,
                "n_from": len(y1),
                "n_to": len(y2),
                "auc_gap": auc_gap,
                "ext_auc": ext["forward"]["auc"],
                "int_auc": auc_int["mean_auc"],
                "geodesic": gdist,
                "centroid_dist": cdist,
            })

    return pd.DataFrame(rows)


def run_loo(df, studies):
    """Leave-one-study-out validation."""
    maes = {"baseline": [], "auc_only": [], "full": []}
    rhos = []
    for held_out in studies:
        test_mask = (df["from_study"] == held_out) | (df["to_study"] == held_out)
        train = df[~test_mask]
        test = df[test_mask]
        if len(train) < 5 or len(test) < 2:
            continue

        mean_gap = train["auc_gap"].mean()
        maes["baseline"].append(np.abs(test["auc_gap"] - mean_gap).mean())

        lr1 = LinearRegression().fit(train[["int_auc"]], train["auc_gap"])
        pred1 = lr1.predict(test[["int_auc"]])
        maes["auc_only"].append(np.abs(test["auc_gap"] - pred1).mean())

        lr2 = LinearRegression().fit(train[["int_auc", "geodesic"]], train["auc_gap"])
        pred2 = lr2.predict(test[["int_auc", "geodesic"]])
        maes["full"].append(np.abs(test["auc_gap"] - pred2).mean())

        rho, _ = spearmanr(pred2, test["auc_gap"])
        rhos.append(rho)

    return {k: float(np.mean(v)) for k, v in maes.items()}, float(np.mean(rhos))


def analyze_disease(disease):
    """Full analysis pipeline for one disease."""
    out_dir = RESULTS_DIR / disease
    out_dir.mkdir(parents=True, exist_ok=True)

    cohorts = load_disease(disease)
    studies = sorted(cohorts.keys())
    C = len(studies)

    if C < 3:
        print(f"  Only {C} studies — need at least 3. Skipping.")
        return

    df = run_ordered_pairs(cohorts, k=K_DEFAULT)
    df.to_csv(out_dir / "ordered_pairs.csv", index=False)

    # Raw and partial correlations
    rho_raw, p_raw = spearmanr(df["geodesic"], df["auc_gap"])
    rho_confound, _ = spearmanr(df["int_auc"], df["auc_gap"])
    rho_partial = partial_spearman(
        df["geodesic"].values, df["auc_gap"].values, df["int_auc"].values
    )

    print(f"\n{'='*60}")
    print(f"RESULTS: {disease.upper()} ({C} studies, {len(df)} pairs)")
    print(f"{'='*60}")
    print(f"  Raw geodesic-gap rho:        {rho_raw:+.3f} (p={p_raw:.4f})")
    print(f"  Source AUC-gap rho:          {rho_confound:+.3f}")
    print(f"  Partial rho (|source AUC):   {rho_partial:+.3f}")

    # Clustered bootstrap
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] Running clustered bootstrap ({N_BOOTSTRAP} iterations)...")
    boot = clustered_bootstrap_partial(df)
    print(f"  Median partial rho: {boot['median']:+.3f}")
    print(f"  95% CI: [{boot['ci_lo']:+.3f}, {boot['ci_hi']:+.3f}]")
    print(f"  % positive: {boot['pct_positive']:.1f}%")

    # OLS delta-R2
    lr_base = LinearRegression().fit(df[["int_auc"]], df["auc_gap"])
    r2_base = lr_base.score(df[["int_auc"]], df["auc_gap"])
    lr_full = LinearRegression().fit(df[["int_auc", "geodesic"]], df["auc_gap"])
    r2_full = lr_full.score(df[["int_auc", "geodesic"]], df["auc_gap"])
    delta_r2 = r2_full - r2_base
    print(f"\n  R2 (source AUC only):  {r2_base:.3f}")
    print(f"  R2 (+ geodesic):       {r2_full:.3f}")
    print(f"  Delta R2:              {delta_r2:+.3f}")

    # k-sensitivity
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] k-sensitivity analysis...")
    k_results = []
    for k in K_RANGE:
        df_k = run_ordered_pairs(cohorts, k=k)
        rho_k, p_k = spearmanr(df_k["geodesic"], df_k["auc_gap"])
        partial_k = partial_spearman(
            df_k["geodesic"].values, df_k["auc_gap"].values, df_k["int_auc"].values
        )
        k_results.append({"k": k, "raw_rho": rho_k, "raw_p": p_k, "partial_rho": partial_k})
        print(f"  k={k:3d}: raw rho={rho_k:+.3f}, partial rho={partial_k:+.3f}")
    pd.DataFrame(k_results).to_csv(out_dir / "k_sensitivity.csv", index=False)

    # LOO validation
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] Leave-one-study-out validation...")
    loo_maes, loo_rho = run_loo(df, studies)
    pct_vs_baseline = (1 - loo_maes["full"] / loo_maes["baseline"]) * 100
    pct_vs_auc = (1 - loo_maes["full"] / loo_maes["auc_only"]) * 100 if loo_maes["auc_only"] > 0 else 0
    print(f"  MAE baseline:   {loo_maes['baseline']:.4f}")
    print(f"  MAE AUC-only:   {loo_maes['auc_only']:.4f}")
    print(f"  MAE full:       {loo_maes['full']:.4f}")
    print(f"  Improvement vs baseline: {pct_vs_baseline:.1f}%")
    print(f"  Improvement vs AUC-only: {pct_vs_auc:.1f}%")
    print(f"  LOO mean rho:   {loo_rho:.3f}")

    # Three-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(df["geodesic"], df["auc_gap"], alpha=0.5, s=20)
    axes[0].set_xlabel("Geodesic distance")
    axes[0].set_ylabel("AUC gap")
    axes[0].set_title(f"Raw: rho={rho_raw:+.2f}")

    axes[1].scatter(df["int_auc"], df["auc_gap"], alpha=0.5, s=20, color="orange")
    axes[1].set_xlabel("Source internal AUC")
    axes[1].set_ylabel("AUC gap")
    axes[1].set_title(f"Confound: rho={rho_confound:+.2f}")

    resid_x = rankdata(df["geodesic"]) - np.polyval(
        np.polyfit(rankdata(df["int_auc"]), rankdata(df["geodesic"]), 1),
        rankdata(df["int_auc"]))
    resid_y = rankdata(df["auc_gap"]) - np.polyval(
        np.polyfit(rankdata(df["int_auc"]), rankdata(df["auc_gap"]), 1),
        rankdata(df["int_auc"]))
    axes[2].scatter(resid_x, resid_y, alpha=0.5, s=20, color="green")
    axes[2].set_xlabel("Geodesic (residualized)")
    axes[2].set_ylabel("AUC gap (residualized)")
    axes[2].set_title(f"Partial: rho={rho_partial:+.2f}")

    fig.suptitle(f"{disease.upper()}: {C} studies, {len(df)} pairs", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "three_panel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved three_panel.png")

    # Save summary
    summary = {
        "disease": disease,
        "n_studies": C,
        "n_pairs": len(df),
        "studies": studies,
        "raw_rho": float(rho_raw),
        "raw_p": float(p_raw),
        "confound_rho": float(rho_confound),
        "partial_rho": float(rho_partial),
        "bootstrap": boot,
        "r2_base": float(r2_base),
        "r2_full": float(r2_full),
        "delta_r2": float(delta_r2),
        "loo": {"maes": loo_maes, "mean_rho": loo_rho},
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary.json")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", choices=["crc", "t2d", "all"], default="all")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    diseases = ["crc", "t2d"] if args.disease == "all" else [args.disease]
    summaries = {}
    for disease in diseases:
        try:
            summaries[disease] = analyze_disease(disease)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

    if len(summaries) > 1:
        print(f"\n{'='*60}")
        print("CROSS-DISEASE SUMMARY")
        print(f"{'='*60}")
        for d, s in summaries.items():
            print(f"  {d.upper():6s}: C={s['n_studies']:2d}, "
                  f"partial rho={s['partial_rho']:+.3f}, "
                  f"CI=[{s['bootstrap']['ci_lo']:+.3f}, {s['bootstrap']['ci_hi']:+.3f}], "
                  f"LOO improvement={((1 - s['loo']['maes']['full']/s['loo']['maes']['baseline'])*100):.0f}%")


if __name__ == "__main__":
    main()

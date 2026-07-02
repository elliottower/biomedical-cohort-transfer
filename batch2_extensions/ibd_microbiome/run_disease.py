"""Generic cross-cohort transportability pipeline for any metagenomics dataset.

Mirrors the CRC pipeline: computes pairwise geodesic distances, AUC gaps,
partial correlations, clustered bootstrap, k-sensitivity, and LOO validation.

Usage:
  PYTHONPATH=. uv run python batch2_extensions/ibd_microbiome/run_disease.py \
    --data-dir batch2_extensions/ibd_microbiome/data \
    --disease IBD \
    --min-prevalence 0.1
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import Counter
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from src.transportability import geodesic_distance, top_k_subspace, centroid_distance
from src.external_validation import external_validation, internal_validation

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


def load_cohorts_from_dir(data_dir, disease_label="disease", control_label="control",
                          min_prevalence=0.1, min_samples=20, pre_transformed=False):
    """Load cohorts from a directory with species_profiles.tsv and metadata.tsv.

    Expected files:
      - species_profiles.tsv: (features x samples) or (samples x features)
      - metadata.tsv: columns include Sample_ID, Study, Group

    Group column should contain disease_label and control_label values.
    """
    data_dir = Path(data_dir)
    print(f"[{datetime.now():%H:%M:%S}] Loading cohorts from {data_dir}...")

    feat_file = data_dir / "species_profiles.tsv"
    meta_file = data_dir / "metadata.tsv"

    if not feat_file.exists():
        for alt in ["abundance.tsv", "profiles.tsv", "features.tsv", "species.tsv"]:
            if (data_dir / alt).exists():
                feat_file = data_dir / alt
                break

    if not meta_file.exists():
        for alt in ["meta.tsv", "sample_metadata.tsv", "meta_all.tsv"]:
            if (data_dir / alt).exists():
                meta_file = data_dir / alt
                break

    if not feat_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            f"Need species_profiles.tsv and metadata.tsv in {data_dir}\n"
            f"Found: {list(data_dir.glob('*.tsv'))}"
        )

    meta = pd.read_csv(meta_file, sep="\t")

    study_col = None
    for col in ["Study", "study", "dataset", "Dataset", "cohort", "Cohort", "study_name"]:
        if col in meta.columns:
            study_col = col
            break
    if study_col is None:
        raise ValueError(f"No study column found. Columns: {list(meta.columns)}")

    group_col = None
    for col in ["Group", "group", "disease", "Disease", "condition", "Condition", "study_condition"]:
        if col in meta.columns:
            group_col = col
            break
    if group_col is None:
        raise ValueError(f"No group column found. Columns: {list(meta.columns)}")

    sample_col = None
    for col in ["Sample_ID", "sample_id", "SampleID", "sample", "Sample"]:
        if col in meta.columns:
            sample_col = col
            break
    if sample_col is None:
        sample_col = meta.columns[0]

    target_groups = [disease_label, control_label]
    meta_filt = meta[meta[group_col].isin(target_groups)].copy()
    meta_filt = meta_filt.set_index(sample_col)
    print(f"  {len(meta_filt)} samples with group in {target_groups}")

    feat = pd.read_csv(feat_file, sep="\t", index_col=0)
    if feat.shape[0] > feat.shape[1]:
        feat = feat.T

    common = feat.columns.intersection(meta_filt.index)
    if len(common) < 50:
        common = feat.index.intersection(meta_filt.index)
        if len(common) > 50:
            feat = feat.T
            common = feat.columns.intersection(meta_filt.index)

    feat = feat[common].T
    meta_filt = meta_filt.loc[common]
    print(f"  {len(common)} samples with features")

    prevalence = (feat > 0).mean(axis=0)
    keep = prevalence >= min_prevalence
    feat = feat.loc[:, keep]
    print(f"  {keep.sum()} features after {min_prevalence:.0%} prevalence filter (dropped {(~keep).sum()})")

    if pre_transformed:
        X_all = feat.values.astype(np.float64)
    else:
        feat_rel = feat.div(feat.sum(axis=1), axis=0)
        X_all = np.log1p(feat_rel.values.astype(np.float64) * 1e6)
    y_all = (meta_filt[group_col] == disease_label).astype(int).values

    studies = sorted(meta_filt[study_col].unique())
    cohorts = {}
    for study in studies:
        mask = (meta_filt[study_col] == study).values
        X = X_all[mask]
        y = y_all[mask]
        if len(y) < min_samples or len(np.unique(y)) < 2:
            print(f"  Skipping {study}: {len(y)} samples, {y.sum()} positive")
            continue
        n_pos = int(y.sum())
        n_neg = int(len(y) - y.sum())
        cohorts[study] = {"X": X, "y": y, "n": len(y), "n_pos": n_pos, "n_neg": n_neg}
        print(f"  {study}: {len(y)} samples ({n_pos} {disease_label}, {n_neg} {control_label})")

    print(f"  {len(cohorts)} studies, {sum(c['n'] for c in cohorts.values())} total samples")
    return cohorts


def compute_all_pairs(cohorts, k=K, seed=SEED):
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
                "geodesic": gdist, "centroid_dist": cdist,
                "auc_gap": gap,
                "int_auc": intv["mean_auc"],
                "ext_auc": ext["forward"]["auc"],
            })
    return pd.DataFrame(rows)


def clustered_bootstrap(df, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    studies = sorted(df["from"].unique())
    n_studies = len(studies)

    pair_idx = {}
    for idx, row in df.iterrows():
        key = (row["from"], row["to"])
        pair_idx.setdefault(key, []).append(idx)

    partials = []
    for _ in tqdm(range(n_boot), desc="Clustered bootstrap"):
        boot_studies = rng.choice(studies, size=n_studies, replace=True)
        study_counts = Counter(boot_studies)

        boot_indices = []
        for (s_from, s_to), idxs in pair_idx.items():
            if s_from in study_counts and s_to in study_counts:
                n_reps = study_counts[s_from] * study_counts[s_to]
                boot_indices.extend(idxs * n_reps)

        if len(boot_indices) < 4:
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
    if len(partials) == 0:
        return {
            "median": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "pct_positive": float("nan"),
            "n_valid": 0,
        }
    return {
        "median": float(np.median(partials)),
        "ci_lo": float(np.percentile(partials, 2.5)),
        "ci_hi": float(np.percentile(partials, 97.5)),
        "pct_positive": float((partials > 0).mean() * 100),
        "n_valid": len(partials),
    }


def k_sensitivity(cohorts, df_template, seed=SEED):
    studies = sorted(cohorts.keys())
    results = []
    for k in [5, 10, 15, 20, 30, 50]:
        rows = []
        for i in range(len(studies)):
            for j in range(len(studies)):
                if i == j:
                    continue
                s1, s2 = studies[i], studies[j]
                X1, X2 = cohorts[s1]["X"], cohorts[s2]["X"]
                U1, _ = top_k_subspace(X1, k)
                U2, _ = top_k_subspace(X2, k)
                rows.append(geodesic_distance(U1, U2))

        geodesics = np.array(rows)
        rho_raw, p_raw = spearmanr(geodesics, df_template["auc_gap"].values)
        rho_partial = partial_spearman(geodesics, df_template["auc_gap"].values, df_template["int_auc"].values)
        results.append({"k": k, "raw_rho": float(rho_raw), "raw_p": float(p_raw), "partial_rho": float(rho_partial)})
        print(f"  k={k}: raw={rho_raw:+.3f}, partial={rho_partial:+.3f}")
    return results


def loo_validation(df):
    studies = sorted(set(df["from"].unique()))
    results = []
    for held_out in studies:
        train = df[(df["from"] != held_out) & (df["to"] != held_out)]
        test = df[(df["from"] == held_out) | (df["to"] == held_out)]
        if len(train) < 10 or len(test) < 2:
            continue

        reg = LinearRegression().fit(train[["int_auc", "geodesic"]].values, train["auc_gap"].values)
        y_pred = reg.predict(test[["int_auc", "geodesic"]].values)

        reg_int = LinearRegression().fit(train[["int_auc"]].values, train["auc_gap"].values)
        y_pred_int = reg_int.predict(test[["int_auc"]].values)

        mae_full = float(np.mean(np.abs(test["auc_gap"].values - y_pred)))
        mae_int = float(np.mean(np.abs(test["auc_gap"].values - y_pred_int)))
        mae_base = float(np.mean(np.abs(test["auc_gap"].values - train["auc_gap"].mean())))
        rho, _ = spearmanr(test["auc_gap"].values, y_pred)

        results.append({
            "held_out": held_out, "mae_full": mae_full,
            "mae_int_only": mae_int, "mae_baseline": mae_base,
            "rho": float(rho),
        })
    return results


def plot_results(df, results, disease, fig_dir):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    geo = df["geodesic"].values
    gap = df["auc_gap"].values
    intauc = df["int_auc"].values

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    rho, p = spearmanr(geo, gap)
    ax.scatter(geo, gap, s=30, c="steelblue", alpha=0.5, edgecolors="navy", linewidth=0.3)
    z = np.polyfit(geo, gap, 1)
    x_line = np.linspace(geo.min() - 0.1, geo.max() + 0.1, 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Geodesic distance")
    ax.set_ylabel("AUC gap")
    ax.set_title(f"Raw: rho = {rho:+.2f}, p = {p:.3f}")

    ax = axes[1]
    rho2, p2 = spearmanr(intauc, gap)
    ax.scatter(intauc, gap, s=30, c="#e6550d", alpha=0.5, edgecolors="#8c2d04", linewidth=0.3)
    z2 = np.polyfit(intauc, gap, 1)
    x2 = np.linspace(intauc.min() - 0.02, intauc.max() + 0.02, 100)
    ax.plot(x2, np.polyval(z2, x2), "--", color="gray", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Source internal AUC")
    ax.set_ylabel("AUC gap")
    ax.set_title(f"Confound: rho = {rho2:+.2f}, p = {p2:.3f}")

    ax = axes[2]
    reg_geo = LinearRegression().fit(intauc.reshape(-1, 1), geo)
    reg_gap = LinearRegression().fit(intauc.reshape(-1, 1), gap)
    resid_geo = geo - reg_geo.predict(intauc.reshape(-1, 1))
    resid_gap = gap - reg_gap.predict(intauc.reshape(-1, 1))
    rho_p = partial_spearman(geo, gap, intauc)
    ax.scatter(resid_geo, resid_gap, s=30, c="#31a354", alpha=0.5, edgecolors="#006d2c", linewidth=0.3)
    z3 = np.polyfit(resid_geo, resid_gap, 1)
    x3 = np.linspace(resid_geo.min() - 0.1, resid_geo.max() + 0.1, 100)
    ax.plot(x3, np.polyval(z3, x3), "--", color="gray", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Geodesic (residual)")
    ax.set_ylabel("AUC gap (residual)")
    ax.set_title(f"Partial: rho = {rho_p:+.2f}")

    fig.suptitle(f"{disease} — geodesic distance vs AUC gap", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / f"F1_{disease.lower()}_partial_correlation.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--disease", default="IBD")
    parser.add_argument("--disease-label", default=None)
    parser.add_argument("--control-label", default=None)
    parser.add_argument("--min-prevalence", type=float, default=0.1)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--pre-transformed", action="store_true",
                        help="Skip relative abundance + log1p (data already transformed)")
    args = parser.parse_args()

    disease_label = args.disease_label or args.disease
    control_label = args.control_label or "control"

    results_dir = Path(args.data_dir).parent / "results"
    fig_dir = Path(args.data_dir).parent / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cohorts = load_cohorts_from_dir(
        args.data_dir, disease_label=disease_label, control_label=control_label,
        min_prevalence=args.min_prevalence, min_samples=args.min_samples,
        pre_transformed=args.pre_transformed,
    )

    if len(cohorts) < 3:
        print(f"Only {len(cohorts)} cohorts found. Need at least 3.")
        return

    print(f"\n[{datetime.now():%H:%M:%S}] Computing {len(cohorts)}×{len(cohorts)-1} ordered pairs...")
    df = compute_all_pairs(cohorts)
    df.to_csv(results_dir / f"{args.disease.lower()}_ordered_pairs.csv", index=False)

    rho_raw, p_raw = spearmanr(df["geodesic"], df["auc_gap"])
    rho_partial = partial_spearman(df["geodesic"].values, df["auc_gap"].values, df["int_auc"].values)
    print(f"\n--- Correlations ---")
    print(f"  Raw:     rho={rho_raw:+.3f}, p={p_raw:.4f}")
    print(f"  Partial: rho={rho_partial:+.3f}")

    print(f"\n[{datetime.now():%H:%M:%S}] Clustered bootstrap...")
    boot = clustered_bootstrap(df)
    if boot['n_valid'] > 0:
        print(f"  Median: {boot['median']:+.3f}, CI [{boot['ci_lo']:+.3f}, {boot['ci_hi']:+.3f}], "
              f"{boot['pct_positive']:.1f}% positive ({boot['n_valid']} valid)")
    else:
        print(f"  Bootstrap: no valid resamples (too few studies)")

    print(f"\n[{datetime.now():%H:%M:%S}] k-sensitivity...")
    ksens = k_sensitivity(cohorts, df)

    print(f"\n[{datetime.now():%H:%M:%S}] LOO validation...")
    loo = loo_validation(df)
    if loo:
        loo_mae_full = np.mean([r["mae_full"] for r in loo])
        loo_mae_base = np.mean([r["mae_baseline"] for r in loo])
        loo_mae_int = np.mean([r["mae_int_only"] for r in loo])
        loo_rho = np.mean([r["rho"] for r in loo])
        print(f"  MAE full={loo_mae_full:.4f}, int-only={loo_mae_int:.4f}, baseline={loo_mae_base:.4f}")
        print(f"  Improvement over baseline: {(1 - loo_mae_full/loo_mae_base)*100:.1f}%")
        print(f"  Mean LOO rho: {loo_rho:.3f}")
    else:
        loo_mae_full = loo_mae_base = loo_mae_int = loo_rho = float("nan")
        print(f"  Skipped: too few studies for LOO")

    all_results = {
        "disease": args.disease,
        "n_studies": len(cohorts),
        "n_pairs": len(df),
        "n_samples": sum(c["n"] for c in cohorts.values()),
        "raw_rho": float(rho_raw),
        "raw_p": float(p_raw),
        "partial_rho": float(rho_partial),
        "clustered_bootstrap": boot,
        "k_sensitivity": ksens,
        "loo_validation": {
            "per_study": loo,
            "mean_mae_full": float(loo_mae_full),
            "mean_mae_int_only": float(loo_mae_int),
            "mean_mae_baseline": float(loo_mae_base),
            "improvement_over_baseline_pct": float((1 - loo_mae_full/loo_mae_base)*100),
            "improvement_over_int_only_pct": float((1 - loo_mae_full/loo_mae_int)*100),
            "mean_rho": float(loo_rho),
        },
    }

    with open(results_dir / f"{args.disease.lower()}_full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[{datetime.now():%H:%M:%S}] Plotting...")
    plot_results(df, all_results, args.disease, fig_dir)

    print(f"\n{'='*60}")
    print(f"{args.disease} SUMMARY")
    print(f"{'='*60}")
    print(f"  Studies:        {len(cohorts)}")
    print(f"  Ordered pairs:  {len(df)}")
    print(f"  Raw rho:        {rho_raw:+.3f} (p={p_raw:.4f})")
    print(f"  Partial rho:    {rho_partial:+.3f}")
    if boot['n_valid'] > 0:
        print(f"  Clustered CI:   [{boot['ci_lo']:+.3f}, {boot['ci_hi']:+.3f}]")
    else:
        print(f"  Clustered CI:   N/A (too few studies)")
    if loo:
        print(f"  LOO MAE:        {loo_mae_full:.4f} ({(1-loo_mae_full/loo_mae_base)*100:.1f}% < baseline)")
        print(f"  LOO rho:        {loo_rho:.3f}")
    else:
        print(f"  LOO:            N/A (too few studies)")


if __name__ == "__main__":
    main()

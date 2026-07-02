"""Cross-cohort transportability on QMDiab metabolomics dataset.

Tests directional transport predictor vs symmetric baselines on ordered
cohort pairs from the Qatar Metabolomics Study on Diabetes.

Cohorts defined by ethnicity (Arab, Filipino, Indian) within each
measurement platform (plasma, urine, saliva). The cross-platform x
ethnicity design gives 9 cohorts = 72 ordered pairs.

Data: Do et al. 2018, Figshare 5904022.

Usage: PYTHONPATH=. uv run python metabolomics-multilab/run_qmdiab.py
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from tqdm import tqdm
from functools import partial

from src.transportability import (
    directional_transport_score, bracket_norm_score,
    geodesic_distance, top_k_subspace, centroid_distance,
)
from src.external_validation import external_validation, internal_validation

DATA_DIR = Path("metabolomics-multilab/data/qmdiab")
RESULTS_DIR = Path("metabolomics-multilab/results")
K = 10
SEED = 42

ETH_MAP = {1: "Arab", 2: "Filipino", 3: "Indian"}
PLATFORMS = ["plasma", "urine", "saliva"]
NON_MET_COLS = {"QMDiab-ID", "AGE", "GENDER", "BMI", "ETH", "T2D"}


def load_qmdiab_cohorts(mode="ethnicity_plasma", min_prevalence=0.1):
    """Load QMDiab metabolomics data, split into cohorts.

    Modes:
        ethnicity_plasma: 3 cohorts by ethnicity on plasma only (501 metabolites)
        cross_platform: 9 cohorts by platform x ethnicity on shared metabolites

    Returns dict: cohort_name -> {"X": array, "y": array, "n": int, ...}
    """
    print(f"[{datetime.now():%H:%M:%S}] Loading QMDiab cohorts (mode={mode})...")

    preprocessed_file = DATA_DIR / "QMDiab_metabolomics_Preprocessed.xlsx"
    if not preprocessed_file.exists():
        raise FileNotFoundError(
            f"Download QMDiab data first from Figshare 5904022.\n"
            f"  Expected: {preprocessed_file}"
        )

    # Load all platforms
    platform_dfs = {}
    for platform in PLATFORMS:
        df = pd.read_excel(preprocessed_file, sheet_name=platform)
        platform_dfs[platform] = df

    if mode == "ethnicity_plasma":
        return _load_ethnicity_cohorts(platform_dfs["plasma"], "plasma", min_prevalence)
    elif mode == "cross_platform":
        return _load_cross_platform_cohorts(platform_dfs, min_prevalence)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _load_ethnicity_cohorts(df, platform_name, min_prevalence):
    """Split a single platform by ethnicity."""
    met_cols = [c for c in df.columns if c not in NON_MET_COLS]
    X_all = df[met_cols].values.astype(np.float64)
    y_all = df["T2D"].values.astype(int)
    eth_all = df["ETH"].values.astype(int)

    # Prevalence filter on the full dataset
    prevalence = (np.isfinite(X_all) & (X_all != 0)).mean(axis=0)
    keep = prevalence >= min_prevalence
    X_all = X_all[:, keep]
    kept_names = [met_cols[i] for i in range(len(met_cols)) if keep[i]]
    print(f"  {keep.sum()} metabolites after {min_prevalence:.0%} prevalence filter "
          f"(dropped {(~keep).sum()})")

    cohorts = {}
    for eth_code, eth_name in ETH_MAP.items():
        mask = eth_all == eth_code
        X = X_all[mask]
        y = y_all[mask]
        name = f"{platform_name}_{eth_name}"

        if len(y) < 20 or len(np.unique(y)) < 2:
            print(f"  Skipping {name}: {len(y)} samples")
            continue

        cohorts[name] = {
            "X": X, "y": y, "n": len(y),
            "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        }
        print(f"  {name}: {len(y)} samples ({y.sum():.0f} T2D, {(1-y).sum():.0f} healthy)")

    print(f"  {len(cohorts)} cohorts, {sum(c['n'] for c in cohorts.values())} total samples, "
          f"{len(kept_names)} features")
    return cohorts


def _load_cross_platform_cohorts(platform_dfs, min_prevalence):
    """Split by platform x ethnicity on shared metabolites."""
    # Find shared metabolites across all platforms
    met_sets = {}
    for platform, df in platform_dfs.items():
        met_sets[platform] = set(df.columns) - NON_MET_COLS
    shared_mets = met_sets["plasma"] & met_sets["urine"] & met_sets["saliva"]
    shared_mets = sorted(shared_mets)
    print(f"  {len(shared_mets)} metabolites shared across all 3 platforms")

    # Prevalence filter on shared metabolites across all data
    all_vals = []
    for df in platform_dfs.values():
        all_vals.append(df[shared_mets].values)
    all_vals = np.vstack(all_vals).astype(np.float64)
    prevalence = (np.isfinite(all_vals) & (all_vals != 0)).mean(axis=0)
    keep = prevalence >= min_prevalence
    kept_mets = [shared_mets[i] for i in range(len(shared_mets)) if keep[i]]
    print(f"  {keep.sum()} metabolites after {min_prevalence:.0%} prevalence filter "
          f"(dropped {(~keep).sum()})")

    cohorts = {}
    for platform, df in platform_dfs.items():
        X_platform = df[kept_mets].values.astype(np.float64)
        y_platform = df["T2D"].values.astype(int)
        eth_platform = df["ETH"].values.astype(int)

        for eth_code, eth_name in ETH_MAP.items():
            mask = eth_platform == eth_code
            X = X_platform[mask]
            y = y_platform[mask]
            name = f"{platform}_{eth_name}"

            if len(y) < 20 or len(np.unique(y)) < 2:
                print(f"  Skipping {name}: {len(y)} samples")
                continue

            cohorts[name] = {
                "X": X, "y": y, "n": len(y),
                "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            }
            print(f"  {name}: {len(y)} samples ({y.sum():.0f} T2D, {(1-y).sum():.0f} healthy)")

    print(f"  {len(cohorts)} cohorts, {sum(c['n'] for c in cohorts.values())} total samples, "
          f"{len(kept_mets)} features")
    return cohorts


def run_ordered_pairs(cohorts, k=K):
    """Run all ordered pairs: directional + symmetric predictors vs AUC gap."""
    print(f"\n[{datetime.now():%H:%M:%S}] Running ordered pairs...")

    names = sorted(cohorts.keys())
    n = len(names)

    rows = []
    for i in tqdm(range(n), desc="Ordered pairs"):
        for j in range(n):
            if i == j:
                continue
            s1, s2 = names[i], names[j]
            X1, y1 = cohorts[s1]["X"], cohorts[s1]["y"]
            X2, y2 = cohorts[s2]["X"], cohorts[s2]["y"]

            # Clamp k to min(n_samples, n_features) - 1
            k_eff = min(k, X1.shape[0] - 1, X2.shape[0] - 1, X1.shape[1] - 1)

            dt = directional_transport_score(X1, y1, X2, y2, k=k_eff)
            bn = bracket_norm_score(X1, y1, X2, y2)

            U1, _ = top_k_subspace(X1, k_eff)
            U2, _ = top_k_subspace(X2, k_eff)
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


def partial_spearman(x, y, z):
    """Partial Spearman correlation between x and y controlling for z.

    Computes rank-based partial correlation:
    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    """
    from scipy.stats import rankdata
    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)

    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]

    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-12:
        return np.nan, np.nan

    r_partial = (r_xy - r_xz * r_yz) / denom

    # Approximate p-value via Fisher z-transform
    n = len(x)
    df = n - 3  # degrees of freedom for partial correlation
    if df < 1:
        return r_partial, np.nan
    from scipy.stats import t as t_dist
    t_stat = r_partial * np.sqrt(df / (1 - r_partial**2 + 1e-12))
    p_val = 2 * t_dist.sf(abs(t_stat), df)
    return float(r_partial), float(p_val)


def print_correlations(df, label=""):
    """Print Spearman and partial Spearman correlations."""
    prefix = f" ({label})" if label else ""
    print(f"\n{'='*70}")
    print(f"SPEARMAN CORRELATIONS WITH AUC GAP{prefix}")
    print(f"{'='*70}")
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
    for col, name in predictors:
        rho, p = spearmanr(df[col], df["auc_gap"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:25s}  {rho:+.3f}  {p:8.4f}  {sig:>4s}")

    print(f"\n  n = {len(df)} ordered pairs")

    # Partial Spearman correlations controlling for source internal AUC
    if len(df) > 5:
        print(f"\n{'='*70}")
        print(f"PARTIAL SPEARMAN (controlling for source internal AUC){prefix}")
        print(f"{'='*70}")
        print(f"  {'Predictor':25s}  {'r_partial':>9s}  {'p':>8s}  {'sig':>4s}")
        print(f"  {'-'*50}")

        for col, name in predictors:
            rp, pp = partial_spearman(
                df[col].values, df["auc_gap"].values, df["int_auc"].values
            )
            sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "" if not np.isnan(pp) else ""
            print(f"  {name:25s}  {rp:+.3f}  {pp:8.4f}  {sig:>4s}")

        print(f"\n  n = {len(df)} ordered pairs, df = {len(df) - 3}")

    # Multivariate model: gap ~ int_auc + geodesic
    if len(df) > 5:
        from sklearn.linear_model import LinearRegression
        X_mv = df[["int_auc", "geodesic"]].values
        y_mv = df["auc_gap"].values
        reg = LinearRegression().fit(X_mv, y_mv)
        r2 = reg.score(X_mv, y_mv)
        print(f"\n  Multivariate R^2 (gap ~ int_auc + geodesic): {r2:.3f}")
        print(f"  Coefficients: int_auc={reg.coef_[0]:.3f}, geodesic={reg.coef_[1]:.3f}, "
              f"intercept={reg.intercept_:.3f}")


def run(mode="cross_platform", k=K, min_prevalence=0.1):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cohorts = load_qmdiab_cohorts(mode=mode, min_prevalence=min_prevalence)
    df = run_ordered_pairs(cohorts, k=k)

    out_prefix = f"qmdiab_{mode}"
    csv_path = RESULTS_DIR / f"{out_prefix}_ordered_pairs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path} ({len(df)} pairs)")

    print_correlations(df, label=mode)

    # Build summary
    summary = {
        "dataset": "QMDiab",
        "mode": mode,
        "n_cohorts": len(set(df["from"])),
        "cohort_names": sorted(set(df["from"])),
        "n_pairs": len(df),
        "k": k,
        "min_prevalence": min_prevalence,
    }

    predictors = ["dt_score", "dt_abs", "dt_alignment", "bn_score", "geodesic", "centroid_dist"]
    for col in predictors:
        rho, p = spearmanr(df[col], df["auc_gap"])
        summary[f"rho_{col}"] = float(rho)
        summary[f"p_{col}"] = float(p)

    if len(df) > 5:
        for col in predictors:
            rp, pp = partial_spearman(
                df[col].values, df["auc_gap"].values, df["int_auc"].values
            )
            summary[f"partial_rho_{col}"] = float(rp)
            summary[f"partial_p_{col}"] = float(pp) if not np.isnan(pp) else None

    json_path = RESULTS_DIR / f"{out_prefix}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{datetime.now():%H:%M:%S}] Done. Results in {RESULTS_DIR}/")
    return df


def run_k_sensitivity(cohorts, k_values=(5, 10, 15, 20, 30, 50)):
    """Run partial correlation at multiple subspace dimensions."""
    print(f"\n[{datetime.now():%H:%M:%S}] Running k-sensitivity...")
    rows = []
    for k in tqdm(k_values, desc="k-sensitivity"):
        df = run_ordered_pairs(cohorts, k=k)
        rho_raw, p_raw = spearmanr(df["geodesic"], df["auc_gap"])
        rho_part, p_part = partial_spearman(
            df["geodesic"].values, df["auc_gap"].values, df["int_auc"].values
        )
        rows.append({
            "k": k, "raw_rho": rho_raw, "raw_p": p_raw,
            "partial_rho": rho_part, "partial_p": p_part,
        })
        print(f"  k={k:3d}  raw={rho_raw:+.3f} (p={p_raw:.4f})  "
              f"partial={rho_part:+.3f} (p={p_part:.4f})")

    return pd.DataFrame(rows)


def run_bootstrap(df, n_boot=1000, seed=42):
    """Bootstrap 95% CI for partial rho(geodesic, gap | int_auc)."""
    print(f"\n[{datetime.now():%H:%M:%S}] Running bootstrap ({n_boot} samples)...")
    rng = np.random.default_rng(seed)
    n = len(df)
    partials = []
    geo = df["geodesic"].values
    gap = df["auc_gap"].values
    intauc = df["int_auc"].values

    for _ in tqdm(range(n_boot), desc="Bootstrap"):
        idx = rng.choice(n, size=n, replace=True)
        rp, _ = partial_spearman(geo[idx], gap[idx], intauc[idx])
        if not np.isnan(rp):
            partials.append(rp)

    partials = np.array(partials)
    ci_lo, ci_hi = np.percentile(partials, [2.5, 97.5])
    median = np.median(partials)
    pct_positive = (partials > 0).mean() * 100

    print(f"  Median partial rho: {median:+.3f}")
    print(f"  95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"  {pct_positive:.1f}% of samples positive")

    return {
        "median": float(median),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "pct_positive": float(pct_positive),
        "n_boot": n_boot,
        "n_valid": len(partials),
    }


def run_robustness(mode="cross_platform", min_prevalence=0.1):
    """Run k-sensitivity and bootstrap on QMDiab."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cohorts = load_qmdiab_cohorts(mode=mode, min_prevalence=min_prevalence)

    # k-sensitivity
    ksens = run_k_sensitivity(cohorts)
    ksens_path = RESULTS_DIR / f"qmdiab_{mode}_k_sensitivity.csv"
    ksens.to_csv(ksens_path, index=False)
    print(f"\n  Saved {ksens_path}")

    # Bootstrap at k=10
    df = run_ordered_pairs(cohorts, k=K)
    boot = run_bootstrap(df)
    boot_path = RESULTS_DIR / f"qmdiab_{mode}_bootstrap.json"
    with open(boot_path, "w") as f:
        json.dump(boot, f, indent=2)
    print(f"  Saved {boot_path}")

    # Combined summary
    summary = {
        "k_sensitivity": ksens.to_dict(orient="records"),
        "bootstrap": boot,
    }
    summary_path = RESULTS_DIR / f"qmdiab_{mode}_robustness.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {summary_path}")

    print(f"\n[{datetime.now():%H:%M:%S}] Robustness analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ethnicity_plasma", "cross_platform"],
                        default="cross_platform")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--min-prevalence", type=float, default=0.1)
    parser.add_argument("--robustness", action="store_true",
                        help="Run k-sensitivity and bootstrap instead of main analysis")
    args = parser.parse_args()
    if args.robustness:
        run_robustness(mode=args.mode, min_prevalence=args.min_prevalence)
    else:
        run(mode=args.mode, k=args.k, min_prevalence=args.min_prevalence)

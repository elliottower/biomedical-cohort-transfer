"""Simulation study: geodesic distance recovers analytical heterogeneity.

Generates synthetic cohorts with controlled feature-level perturbations
(analytical heterogeneity) and verifies that the partial-correlation
pipeline recovers the geodesic-AUC-gap relationship.

The generative model: each cohort observes a shared k-dimensional latent
process Z through a cohort-specific mixing matrix A_c = A_shared + h*E_c.
The perturbation scale h controls analytical heterogeneity. Labels depend
on Z @ beta (shared across cohorts), so a classifier trained on cohort i
degrades on cohort j when A_i differs from A_j.

Usage:
    PYTHONPATH=. uv run python expansion/simulation.py
    PYTHONPATH=. uv run python expansion/simulation.py --n-cohorts 20 --n-reps 200
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*constant.*")

from src.transportability import geodesic_distance, top_k_subspace

RESULTS_DIR = Path("expansion/results/simulation")
SEED_BASE = 12345


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-12:
        return np.nan
    return (r_xy - r_xz * r_yz) / denom


def generate_cohort(n_samples, d, k, beta_true, A_shared, perturbation_scale,
                    obs_noise_std, rng):
    """Generate a synthetic cohort with perturbed mixing matrix.

    X = Z @ A_c + noise, where A_c = A_shared + perturbation_scale * E.
    Labels: y ~ Bernoulli(sigmoid(Z @ beta_true)).
    """
    E = rng.standard_normal((k, d))
    A_c = A_shared + perturbation_scale * E

    Z = rng.standard_normal((n_samples, k))
    noise = rng.standard_normal((n_samples, d)) * obs_noise_std
    X = Z @ A_c + noise

    logits = Z @ beta_true
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.random(n_samples) < probs).astype(int)

    return X, y


def internal_auc(X, y, seed):
    """5-fold cross-validated AUC."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        lr = LogisticRegression(C=1, max_iter=1000, random_state=seed)
        lr.fit(X[train_idx], y[train_idx])
        proba = lr.predict_proba(X[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[test_idx], proba))
    return np.mean(aucs) if aucs else 0.5


def external_auc(X_train, y_train, X_test, y_test, seed):
    """Train on one cohort, test on another."""
    lr = LogisticRegression(C=1, max_iter=1000, random_state=seed)
    lr.fit(X_train, y_train)
    proba = lr.predict_proba(X_test)[:, 1]
    if len(np.unique(y_test)) < 2:
        return 0.5
    return roc_auc_score(y_test, proba)


def run_simulation(n_cohorts, d, k, n_per_cohort, heterogeneity_levels,
                   obs_noise_std, n_reps, seed):
    """Run simulation across heterogeneity levels."""
    rng = np.random.default_rng(seed)

    A_shared = rng.standard_normal((k, d)) * 0.3
    beta_true = rng.standard_normal(k)
    beta_true = beta_true / np.linalg.norm(beta_true) * 3.0

    results = []

    for h in tqdm(heterogeneity_levels, desc="Heterogeneity levels"):
        for rep in range(n_reps):
            rep_rng = np.random.default_rng(seed + rep + int(h * 1000) * 1000000)

            pert_scales = rep_rng.uniform(0, h, size=n_cohorts)

            cohorts = []
            for c in range(n_cohorts):
                X, y = generate_cohort(
                    n_per_cohort, d, k, beta_true, A_shared,
                    pert_scales[c], obs_noise_std, rep_rng,
                )
                cohorts.append({"X": X, "y": y, "pert": pert_scales[c]})

            rows = []
            for i in range(n_cohorts):
                for j in range(n_cohorts):
                    if i == j:
                        continue
                    U1, _ = top_k_subspace(cohorts[i]["X"], k)
                    U2, _ = top_k_subspace(cohorts[j]["X"], k)
                    gdist = geodesic_distance(U1, U2)

                    auc_int = internal_auc(cohorts[i]["X"], cohorts[i]["y"], seed=rep)
                    auc_ext = external_auc(
                        cohorts[i]["X"], cohorts[i]["y"],
                        cohorts[j]["X"], cohorts[j]["y"],
                        seed=rep,
                    )
                    auc_gap = auc_int - auc_ext

                    rows.append({
                        "geodesic": gdist,
                        "auc_gap": auc_gap,
                        "int_auc": auc_int,
                        "true_pert_diff": abs(pert_scales[i] - pert_scales[j]),
                    })

            df = np.array([(r["geodesic"], r["auc_gap"], r["int_auc"]) for r in rows])
            if len(df) < 6:
                continue

            rho_raw, _ = spearmanr(df[:, 0], df[:, 1])
            rho_partial = partial_spearman(df[:, 0], df[:, 1], df[:, 2])

            rho_recovery, _ = spearmanr(
                [r["geodesic"] for r in rows],
                [r["true_pert_diff"] for r in rows],
            )

            results.append({
                "heterogeneity": h,
                "rep": rep,
                "n_cohorts": n_cohorts,
                "raw_rho": float(rho_raw),
                "partial_rho": float(rho_partial),
                "pert_recovery_rho": float(rho_recovery),
                "mean_geodesic": float(np.mean(df[:, 0])),
                "mean_gap": float(np.mean(df[:, 1])),
            })

    return pd.DataFrame(results) if results else None


def run_power_curve(d, k, n_per_cohort, heterogeneity, obs_noise_std,
                    cohort_counts, n_reps, seed):
    """Power curve: detectable effect size vs number of cohorts."""
    rng = np.random.default_rng(seed)
    A_shared = rng.standard_normal((k, d)) * 0.3
    beta_true = rng.standard_normal(k)
    beta_true = beta_true / np.linalg.norm(beta_true) * 3.0

    results = []
    for C in tqdm(cohort_counts, desc="Cohort counts"):
        partials = []
        for rep in range(n_reps):
            rep_rng = np.random.default_rng(seed + rep + C * 1000000)
            pert_scales = rep_rng.uniform(0, heterogeneity, size=C)

            cohorts = []
            for c in range(C):
                X, y = generate_cohort(
                    n_per_cohort, d, k, beta_true, A_shared,
                    pert_scales[c], obs_noise_std, rep_rng,
                )
                cohorts.append({"X": X, "y": y})

            rows = []
            for i in range(C):
                for j in range(C):
                    if i == j:
                        continue
                    U1, _ = top_k_subspace(cohorts[i]["X"], k)
                    U2, _ = top_k_subspace(cohorts[j]["X"], k)
                    gdist = geodesic_distance(U1, U2)
                    auc_int = internal_auc(cohorts[i]["X"], cohorts[i]["y"], seed=rep)
                    auc_ext = external_auc(
                        cohorts[i]["X"], cohorts[i]["y"],
                        cohorts[j]["X"], cohorts[j]["y"], seed=rep,
                    )
                    rows.append({"geodesic": gdist, "auc_gap": auc_int - auc_ext, "int_auc": auc_int})

            arr = np.array([(r["geodesic"], r["auc_gap"], r["int_auc"]) for r in rows])
            rho = partial_spearman(arr[:, 0], arr[:, 1], arr[:, 2])
            partials.append(rho)

        partials = [p for p in partials if not np.isnan(p)]
        power = np.mean([abs(p) > 0.3 for p in partials]) if partials else 0
        results.append({
            "n_cohorts": C,
            "mean_partial_rho": float(np.mean(partials)) if partials else 0,
            "std_partial_rho": float(np.std(partials)) if partials else 0,
            "power_at_03": float(power),
            "n_valid": len(partials),
        })
        print(f"  C={C:3d}: mean partial rho={results[-1]['mean_partial_rho']:+.3f}, "
              f"power={power:.2f}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cohorts", type=int, default=12)
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--d", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-per-cohort", type=int, default=150)
    parser.add_argument("--obs-noise", type=float, default=0.1)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H:%M:%S")

    # Part 1: Heterogeneity sweep
    print(f"[{ts}] Part 1: Heterogeneity sweep (C={args.n_cohorts}, {args.n_reps} reps)")
    h_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    df_h = run_simulation(
        n_cohorts=args.n_cohorts, d=args.d, k=args.k,
        n_per_cohort=args.n_per_cohort,
        heterogeneity_levels=h_levels,
        obs_noise_std=args.obs_noise, n_reps=args.n_reps, seed=SEED_BASE,
    )
    if df_h is not None:
        df_h.to_csv(RESULTS_DIR / "heterogeneity_sweep.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for metric, ax, title in [
            ("partial_rho", axes[0], "Partial correlation"),
            ("pert_recovery_rho", axes[1], "Perturbation recovery"),
        ]:
            means = df_h.groupby("heterogeneity")[metric].mean()
            stds = df_h.groupby("heterogeneity")[metric].std()
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        fmt="o-", capsize=4)
            ax.set_xlabel("Heterogeneity level")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(title)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        plt.suptitle(f"Simulation: C={args.n_cohorts}, n={args.n_per_cohort}, "
                     f"d={args.d}, k={args.k}", fontsize=13)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "heterogeneity_sweep.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved heterogeneity_sweep.png")

    # Part 2: Power curve
    ts = datetime.now().strftime("%H:%M:%S")
    cohort_counts = [5, 7, 9, 12, 15, 20, 25]
    print(f"\n[{ts}] Part 2: Power curve (h=0.3, {min(args.n_reps, 50)} reps)")
    df_power = run_power_curve(
        d=args.d, k=args.k, n_per_cohort=args.n_per_cohort,
        heterogeneity=0.3, obs_noise_std=args.obs_noise,
        cohort_counts=cohort_counts,
        n_reps=min(args.n_reps, 50), seed=SEED_BASE + 99999,
    )
    df_power.to_csv(RESULTS_DIR / "power_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_power["n_cohorts"], df_power["power_at_03"], "o-", linewidth=2)
    ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="80% power")
    ax.set_xlabel("Number of cohorts")
    ax.set_ylabel("Power (|partial rho| > 0.3)")
    ax.set_title("Power curve: detectable effect vs number of studies")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "power_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved power_curve.png")

    # Summary
    summary = {
        "params": vars(args),
        "heterogeneity_levels": h_levels,
        "cohort_counts": cohort_counts,
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{datetime.now():%H:%M:%S}] Done. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

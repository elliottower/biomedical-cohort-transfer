"""Generate QMDiab figures for paper_v2b.

Usage: PYTHONPATH=. uv run python metabolomics-multilab/make_qmdiab_figures.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

RESULTS_DIR = Path("metabolomics-multilab/results")
FIG_DIR = RESULTS_DIR / "figures"


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def partial_spearman(x, y, z):
    from scipy.stats import rankdata, t as t_dist
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-12:
        return np.nan, np.nan
    r_partial = (r_xy - r_xz * r_yz) / denom
    df = len(x) - 3
    t_stat = r_partial * np.sqrt(df / (1 - r_partial**2 + 1e-12))
    p_val = 2 * t_dist.sf(abs(t_stat), df)
    return float(r_partial), float(p_val)


def plot_partial_correlation(df, outpath):
    """3-panel figure: raw scatter, confound, partial residuals."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    geo = df["geodesic"].values
    gap = df["auc_gap"].values
    intauc = df["int_auc"].values

    # Panel 1: raw scatter
    ax = axes[0]
    rho, p = spearmanr(geo, gap)
    ax.scatter(geo, gap, s=40, c="steelblue", alpha=0.6, edgecolors="navy", linewidth=0.4)
    z = np.polyfit(geo, gap, 1)
    x_line = np.linspace(geo.min() - 0.1, geo.max() + 0.1, 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Geodesic distance")
    ax.set_ylabel("AUC gap")
    ax.set_title(f"Raw: rho = {rho:+.2f}, p = {p:.3f}")

    # Panel 2: confound (int_auc vs gap)
    ax = axes[1]
    rho2, p2 = spearmanr(intauc, gap)
    ax.scatter(intauc, gap, s=40, c="#e6550d", alpha=0.6, edgecolors="#8c2d04", linewidth=0.4)
    z2 = np.polyfit(intauc, gap, 1)
    x2 = np.linspace(intauc.min() - 0.02, intauc.max() + 0.02, 100)
    ax.plot(x2, np.polyval(z2, x2), "--", color="gray", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Source internal AUC")
    ax.set_ylabel("AUC gap")
    ax.set_title(f"Confound: rho = {rho2:+.2f}, p = {p2:.3f}")

    # Panel 3: partial residuals
    ax = axes[2]
    from sklearn.linear_model import LinearRegression
    reg_geo = LinearRegression().fit(intauc.reshape(-1, 1), geo)
    reg_gap = LinearRegression().fit(intauc.reshape(-1, 1), gap)
    resid_geo = geo - reg_geo.predict(intauc.reshape(-1, 1))
    resid_gap = gap - reg_gap.predict(intauc.reshape(-1, 1))

    rho_p, p_p = partial_spearman(geo, gap, intauc)
    ax.scatter(resid_geo, resid_gap, s=40, c="#31a354", alpha=0.6, edgecolors="#006d2c", linewidth=0.4)
    z3 = np.polyfit(resid_geo, resid_gap, 1)
    x3 = np.linspace(resid_geo.min() - 0.1, resid_geo.max() + 0.1, 100)
    ax.plot(x3, np.polyval(z3, x3), "--", color="gray", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Geodesic (residual)")
    ax.set_ylabel("AUC gap (residual)")
    p_str = f"{p_p:.4f}" if p_p >= 0.0001 else "<0.0001"
    ax.set_title(f"Partial: rho = {rho_p:+.2f}, p = {p_str}")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_k_sensitivity(robustness, outpath):
    """k-sensitivity: raw and partial rho across k values."""
    set_style()
    ks_data = robustness["k_sensitivity"]
    ks = [d["k"] for d in ks_data]
    raw = [d["raw_rho"] for d in ks_data]
    partial = [d["partial_rho"] for d in ks_data]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, raw, "o-", color="steelblue", linewidth=2, markersize=7, label="Raw Spearman")
    ax.plot(ks, partial, "s-", color="#31a354", linewidth=2, markersize=7, label="Partial (| int. AUC)")

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Subspace dimension $k$")
    ax.set_ylabel("Spearman rho with AUC gap")
    ax.set_title("QMDiab: k-sensitivity of geodesic--gap correlation")
    ax.legend(frameon=False)
    ax.set_xticks(ks)
    ax.set_ylim(-0.1, 0.7)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_DIR / "qmdiab_cross_platform_ordered_pairs.csv")
    plot_partial_correlation(df, FIG_DIR / "F1_qmdiab_partial_correlation.png")

    with open(RESULTS_DIR / "qmdiab_cross_platform_robustness.json") as f:
        robustness = json.load(f)
    plot_k_sensitivity(robustness, FIG_DIR / "F2_qmdiab_k_sensitivity.png")

    print("\nDone.")

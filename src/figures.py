"""Publication-quality figure generation for geometric transportability.

Style reference: paper_curvature_v2.tex figures — KDE with shaded tails,
annotated heatmaps, labeled scatters, diverging colormaps.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde, spearmanr
from pathlib import Path


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
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def plot_geom_vs_gap(geom_scores, auc_gaps, labels, outpath, title=None):
    """F1: Geodesic distance vs AUC gap scatter with regression line."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 5.5))

    geom_scores = np.asarray(geom_scores)
    auc_gaps = np.asarray(auc_gaps)

    ax.scatter(geom_scores, auc_gaps, s=45, c="steelblue", alpha=0.5,
               edgecolors="navy", linewidth=0.4, zorder=3)

    if len(geom_scores) > 2:
        rho, p = spearmanr(geom_scores, auc_gaps)
        z = np.polyfit(geom_scores, auc_gaps, 1)
        x_line = np.linspace(geom_scores.min() - 0.05, geom_scores.max() + 0.05, 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="gray",
                alpha=0.7, linewidth=1.5, label=f"$\\rho = {rho:.3f}$, $p = {p:.3f}$")
        ax.legend(frameon=False, loc="upper left")

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Grassmannian geodesic distance")
    ax.set_ylabel("AUC gap (internal $-$ external)")
    if title:
        ax.set_title(title)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_principal_angles(angles_list, labels, outpath):
    """F2: Principal angle spectrum — shaded area + line."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = ["#2171b5", "#cb181d", "#238b45", "#6a51a3"]
    for i, (angles, label) in enumerate(zip(angles_list, labels)):
        angles_deg = np.degrees(angles)
        idx = np.arange(1, len(angles) + 1)
        c = colors[i % len(colors)]
        ax.fill_between(idx, 0, angles_deg, alpha=0.15, color=c)
        ax.plot(idx, angles_deg, "o-", color=c, label=label,
                markersize=6, linewidth=2, markeredgecolor="white", markeredgewidth=0.5)

    ax.axhline(90, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xlabel("Principal angle index $i$")
    ax.set_ylabel("$\\theta_i$ (degrees)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_null_distributions(observed, null_dist, score_name, outpath):
    """F3: KDE null distribution with shaded tails and observed line."""
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))

    null_dist = np.asarray(null_dist, dtype=float)
    observed = float(observed)

    kde = gaussian_kde(null_dist, bw_method=0.3)
    x_grid = np.linspace(null_dist.min() - 0.1 * np.ptp(null_dist),
                         max(null_dist.max(), observed) + 0.1 * np.ptp(null_dist), 300)
    density = kde(x_grid)

    ax.fill_between(x_grid, density, alpha=0.3, color="#6baed6")
    ax.plot(x_grid, density, color="#2171b5", linewidth=1.5)

    tail_mask = x_grid >= observed
    ax.fill_between(x_grid[tail_mask], density[tail_mask], alpha=0.4, color="#fcae91")

    ax.axvline(observed, color="#cb181d", linewidth=2.5, zorder=5)
    ax.text(observed, ax.get_ylim()[1] * 0.92, f"  obs = {observed:.3f}",
            color="#cb181d", fontsize=10, va="top", fontweight="bold")

    p = np.mean(null_dist >= observed)
    z = (observed - null_dist.mean()) / (null_dist.std() + 1e-12)
    ax.set_title(f"{score_name}  ($z = {z:+.2f}$, $p = {p:.4f}$, "
                 f"$n = {len(null_dist):,}$)")
    ax.set_xlabel(score_name)
    ax.set_ylabel("Density")
    ax.set_yticks([])

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_null_grid(panels, outpath, ncols=3):
    """Multi-panel null distribution grid (like curvature fig5).

    panels: list of dicts with keys:
        observed, null_dist, title (short label)
    """
    set_style()
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    axes_flat = axes.flatten()

    for i, panel in enumerate(panels):
        ax = axes_flat[i]
        null = np.asarray(panel["null_dist"], dtype=float)
        obs = float(panel["observed"])

        kde = gaussian_kde(null, bw_method=0.3)
        xg = np.linspace(null.min() - 0.1 * np.ptp(null),
                         max(null.max(), obs) + 0.1 * np.ptp(null), 300)
        dens = kde(xg)

        ax.fill_between(xg, dens, alpha=0.3, color="#6baed6")
        ax.plot(xg, dens, color="#2171b5", linewidth=1.2)

        tail = xg >= obs
        ax.fill_between(xg[tail], dens[tail], alpha=0.4, color="#fcae91")
        ax.axvline(obs, color="#cb181d", linewidth=2)

        z = (obs - null.mean()) / (null.std() + 1e-12)
        sig = "*" if abs(z) > 1.96 else ""
        ax.set_title(f"{panel['title']}  ($z = {z:+.1f}{sig}$)", fontsize=11)
        ax.text(obs, ax.get_ylim()[1] * 0.88, f"  obs = {obs:.3f}",
                color="#cb181d", fontsize=8, va="top")
        ax.set_ylabel("Density")
        ax.set_yticks([])

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_distance_heatmap(dist_matrix, labels, outpath, annotate=True):
    """Pairwise geodesic distance heatmap with annotations."""
    set_style()
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.7 * n + 2, 0.6 * n + 1.5))

    off_diag = dist_matrix[~np.eye(n, dtype=bool)]
    vmin_data = np.min(off_diag)
    vmax_data = np.max(off_diag)
    pad = 0.05 * (vmax_data - vmin_data)
    display = dist_matrix.copy()
    np.fill_diagonal(display, np.nan)
    im = ax.imshow(display, cmap="YlOrRd", vmin=vmin_data - pad,
                   vmax=vmax_data + pad, aspect="auto")

    mid = (vmin_data + vmax_data) / 2
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = dist_matrix[i, j]
            color = "white" if val > mid else "black"
            if annotate and n <= 16:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7 if n > 10 else 9, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Pairwise Grassmannian geodesic distance")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Geodesic distance", fontsize=10)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_zscore_heatmap(zscore_matrix, row_labels, col_labels, outpath,
                        title="Edge $z$-scores vs permutation null",
                        cbar_label=None):
    """Annotated z-score heatmap with diverging colormap (like curvature fig3)."""
    set_style()
    nr, nc = zscore_matrix.shape
    fig, ax = plt.subplots(figsize=(0.9 * nc + 2, 0.6 * nr + 1.5))

    zmax = max(abs(np.nanmin(zscore_matrix)), abs(np.nanmax(zscore_matrix)), 1.0)
    norm = TwoSlopeNorm(vmin=-zmax, vcenter=0, vmax=zmax)

    masked = np.ma.masked_invalid(zscore_matrix)
    im = ax.imshow(masked, cmap="RdBu_r", norm=norm, aspect="auto")

    for i in range(nr):
        for j in range(nc):
            val = zscore_matrix[i, j]
            if np.isnan(val):
                continue
            stars = ""
            if abs(val) > 2.58:
                stars = "***"
            elif abs(val) > 1.96:
                stars = "**"
            elif abs(val) > 1.645:
                stars = "*"
            color = "white" if abs(val) > 0.6 * zmax else "black"
            ax.text(j, i, f"{val:+.1f}{stars}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if stars else "normal")

    ax.set_xticks(range(nc))
    ax.set_yticks(range(nr))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=13)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label or "$z$-score (vs permutation null)", fontsize=10)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_holonomy_bar(subcycles_df, outpath, sort=True):
    """Bar chart of holonomy norms by subcycle (like curvature fig6)."""
    set_style()
    df = subcycles_df.copy()
    if sort:
        df = df.sort_values("holonomy_norm", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(len(df) * 0.8 + 2, 6), 4.5))

    labels = df["cycle"].values
    values = df["holonomy_norm"].values
    sizes = df["size"].values

    cmap = plt.cm.YlGn
    norm_vals = (values - values.min()) / (values.max() - values.min() + 1e-12)
    colors = [cmap(0.3 + 0.6 * v) for v in norm_vals]

    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="black",
                  linewidth=0.5, width=0.7)

    for i, (v, s) in enumerate(zip(values, sizes)):
        ax.text(i, v + 0.02 * values.max(), f"$k={s}$", ha="center",
                fontsize=8, color="gray")

    short_labels = []
    for lab in labels:
        plates = lab.split(" -> ")
        if len(plates) <= 3:
            short_labels.append(lab.replace(" -> ", "\n"))
        else:
            short_labels.append(f"{plates[0]}\n...\n{plates[-1]}")

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(short_labels, fontsize=7, rotation=0)
    ax.set_ylabel("Holonomy norm $\\|\\Phi - I\\|_F$")
    ax.set_title("Cocycle holonomy by subcycle")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_roc_comparison(y_true_int, prob_int, y_true_ext, prob_ext, outpath):
    """ROC curves for internal vs external validation."""
    from sklearn.metrics import roc_curve, roc_auc_score
    set_style()
    fig, ax = plt.subplots(figsize=(5.5, 5))

    fpr_i, tpr_i, _ = roc_curve(y_true_int, prob_int)
    auc_i = roc_auc_score(y_true_int, prob_int)
    ax.plot(fpr_i, tpr_i, color="#2171b5", linewidth=2,
            label=f"Internal (AUC = {auc_i:.3f})")

    fpr_e, tpr_e, _ = roc_curve(y_true_ext, prob_ext)
    auc_e = roc_auc_score(y_true_ext, prob_ext)
    ax.plot(fpr_e, tpr_e, color="#cb181d", linewidth=2,
            label=f"External (AUC = {auc_e:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal")

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_explained_variance(ev_list, labels, outpath):
    """Explained variance by PC index for each cohort."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = ["#2171b5", "#cb181d", "#238b45", "#6a51a3"]
    for i, (ev, label) in enumerate(zip(ev_list, labels)):
        c = colors[i % len(colors)]
        cumev = np.cumsum(ev)
        ax.bar(np.arange(1, len(ev) + 1) + i * 0.25 - 0.125,
               ev, width=0.22, color=c, alpha=0.7, label=f"{label} (cum={cumev[-1]:.1%})")

    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Explained variance ratio")
    ax.legend(frameon=False, fontsize=9)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved {outpath}")

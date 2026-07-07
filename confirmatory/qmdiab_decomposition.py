"""
QMDiab decomposition: split partial rho by pair type.

Cross-biofluid pairs vs within-biofluid cross-ethnicity pairs.
"""

import csv
import numpy as np
from scipy import stats


def partial_spearman(x, y, z):
    """Partial Spearman correlation between x and y, controlling for z."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)

    rho_xy = np.corrcoef(rx, ry)[0, 1]
    rho_xz = np.corrcoef(rx, rz)[0, 1]
    rho_yz = np.corrcoef(ry, rz)[0, 1]

    num = rho_xy - rho_xz * rho_yz
    den = np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
    if den == 0:
        return float('nan')
    return num / den


rows = []
with open('metabolomics-multilab/results/qmdiab_cross_platform_ordered_pairs.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Total pairs: {len(rows)}")
print(f"Sample pair: from={rows[0]['from']}, to={rows[0]['to']}")
print()

cross_biofluid = []
within_biofluid_cross_ethnicity = []
within_biofluid_within_ethnicity = []

for row in rows:
    src_parts = row['from'].split('_')
    tgt_parts = row['to'].split('_')
    src_fluid = src_parts[0]
    src_eth = src_parts[1]
    tgt_fluid = tgt_parts[0]
    tgt_eth = tgt_parts[1]

    if src_fluid != tgt_fluid:
        cross_biofluid.append(row)
    elif src_eth != tgt_eth:
        within_biofluid_cross_ethnicity.append(row)
    else:
        within_biofluid_within_ethnicity.append(row)

print(f"Cross-biofluid pairs: {len(cross_biofluid)}")
print(f"Within-biofluid cross-ethnicity pairs: {len(within_biofluid_cross_ethnicity)}")
print(f"Within-biofluid within-ethnicity pairs: {len(within_biofluid_within_ethnicity)}")
print()

for label, subset in [
    ("ALL PAIRS", rows),
    ("CROSS-BIOFLUID", cross_biofluid),
    ("WITHIN-BIOFLUID CROSS-ETHNICITY", within_biofluid_cross_ethnicity),
]:
    if len(subset) < 5:
        print(f"{label} ({len(subset)} pairs): too few for correlation")
        continue

    geodesic = np.array([float(r['geodesic']) for r in subset])
    gap = np.array([float(r['auc_gap']) for r in subset])
    int_auc = np.array([float(r['int_auc']) for r in subset])

    raw_rho, raw_p = stats.spearmanr(geodesic, gap)
    partial_rho = partial_spearman(geodesic, gap, int_auc)

    mean_gap = np.mean(gap)
    mean_geodesic = np.mean(geodesic)
    std_gap = np.std(gap)

    print(f"{label} ({len(subset)} pairs):")
    print(f"  Mean AUC gap: {mean_gap:+.3f} (SD {std_gap:.3f})")
    print(f"  Mean geodesic: {mean_geodesic:.3f}")
    print(f"  Raw rho: {raw_rho:+.3f} (p={raw_p:.4f})")
    print(f"  Partial rho (| int AUC): {partial_rho:+.3f}")
    print()

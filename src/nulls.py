"""Permutation null models for geometric transportability scores.

Three null models:
1. Permute cohort labels -> H^1 / principal-angle null
2. Permute clinical labels -> AUC gap null
3. Dimension-matched control -> geometric score isn't tracking #features
"""
import numpy as np
from tqdm import tqdm

from src.transportability import sheaf_h1_two_cohort, geodesic_distance, top_k_subspace
from src.external_validation import external_validation


def null_cohort_labels(X_pooled, cohort_labels, k, n_permutations=1000, seed=42):
    """Permute cohort assignments, recompute geometric scores.

    X_pooled: (n_total x features) combined data
    cohort_labels: array of 0/1 indicating which cohort each sample belongs to

    Returns dict with null distributions and z-scores/p-values for observed.
    """
    rng = np.random.default_rng(seed)

    mask1 = cohort_labels == 0
    mask2 = cohort_labels == 1
    observed = sheaf_h1_two_cohort(X_pooled[mask1], X_pooled[mask2], k)

    null_geodesic = []
    null_holonomy = []

    for _ in tqdm(range(n_permutations), desc="  null: cohort labels"):
        perm = rng.permutation(len(cohort_labels))
        perm_labels = cohort_labels[perm]
        m1 = perm_labels == 0
        m2 = perm_labels == 1
        if m1.sum() < k + 1 or m2.sum() < k + 1:
            continue
        result = sheaf_h1_two_cohort(X_pooled[m1], X_pooled[m2], k)
        null_geodesic.append(result["geodesic_dist"])
        null_holonomy.append(result["holonomy_norm"])

    null_geodesic = np.array(null_geodesic)
    null_holonomy = np.array(null_holonomy)

    geo_z = (observed["geodesic_dist"] - null_geodesic.mean()) / null_geodesic.std() if null_geodesic.std() > 0 else 0.0
    hol_z = (observed["holonomy_norm"] - null_holonomy.mean()) / null_holonomy.std() if null_holonomy.std() > 0 else 0.0

    geo_p = float(np.mean(null_geodesic >= observed["geodesic_dist"]))
    hol_p = float(np.mean(null_holonomy >= observed["holonomy_norm"]))

    return {
        "observed_geodesic": observed["geodesic_dist"],
        "observed_holonomy": observed["holonomy_norm"],
        "null_geodesic": null_geodesic,
        "null_holonomy": null_holonomy,
        "geodesic_z": float(geo_z),
        "holonomy_z": float(hol_z),
        "geodesic_p": geo_p,
        "holonomy_p": hol_p,
    }


def null_clinical_labels(X1, y1, X2, y2, classifier="logistic", n_permutations=1000, seed=42):
    """Permute clinical labels within each cohort, recompute external AUC.

    Returns null distribution of AUC gap (internal - external).
    """
    rng = np.random.default_rng(seed)

    from src.external_validation import internal_validation
    real_internal = internal_validation(X1, y1, classifier=classifier, seed=seed)
    real_external = external_validation(X1, y1, X2, y2, classifier=classifier, n_bootstrap=0, seed=seed)
    real_gap = real_internal["mean_auc"] - real_external["forward"]["auc"]

    null_gaps = []
    for _ in tqdm(range(n_permutations), desc="  null: clinical labels"):
        y1_perm = rng.permutation(y1)
        y2_perm = rng.permutation(y2)
        try:
            int_result = internal_validation(X1, y1_perm, classifier=classifier, seed=seed)
            ext_result = external_validation(X1, y1_perm, X2, y2_perm, classifier=classifier, n_bootstrap=0, seed=seed)
            null_gaps.append(int_result["mean_auc"] - ext_result["forward"]["auc"])
        except Exception:
            continue

    null_gaps = np.array(null_gaps)
    gap_z = (real_gap - null_gaps.mean()) / null_gaps.std() if null_gaps.std() > 0 else 0.0
    gap_p = float(np.mean(null_gaps >= real_gap))

    return {
        "observed_gap": float(real_gap),
        "null_gaps": null_gaps,
        "gap_z": float(gap_z),
        "gap_p": gap_p,
    }

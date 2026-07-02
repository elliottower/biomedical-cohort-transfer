"""Preprocessing: align features across cohorts, log-transform, extract metadata.

Takes raw downloaded data and produces aligned feature matrices per cohort
in data/processed/.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def load_feature_matrix(path):
    """Load a feature matrix from CSV/TSV/Excel.

    Expects samples as rows, features as columns.
    First column may be sample ID; looks for a 'label' or 'class' column.
    """
    path = Path(path)
    if path.suffix in (".csv", ".txt", ".tsv"):
        sep = "\t" if path.suffix in (".tsv", ".txt") else ","
        df = pd.read_csv(path, sep=sep)
    elif path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return df


def align_features(df1, df2, feature_cols=None):
    """Align feature columns across two cohort dataframes.

    Returns (X1, X2) as numpy arrays with the same feature set (intersection).
    """
    if feature_cols is None:
        meta_cols = {"sample", "id", "sample_id", "label", "class", "group",
                     "cohort", "batch", "plate", "site", "subject", "patient"}
        feat1 = [c for c in df1.columns if c.lower() not in meta_cols]
        feat2 = [c for c in df2.columns if c.lower() not in meta_cols]
        common = sorted(set(feat1) & set(feat2))
    else:
        common = feature_cols

    if len(common) == 0:
        raise ValueError("No overlapping features found between cohorts")

    X1 = df1[common].values.astype(np.float64)
    X2 = df2[common].values.astype(np.float64)

    return X1, X2, common


def extract_labels(df, label_col=None):
    """Extract binary clinical labels from a dataframe.

    Tries common column names if label_col is not specified.
    """
    if label_col is not None:
        return df[label_col].values

    candidates = ["label", "class", "group", "diagnosis", "condition",
                  "status", "outcome", "Label", "Class", "Group"]
    for col in candidates:
        if col in df.columns:
            vals = df[col].values
            unique = np.unique(vals)
            if len(unique) == 2:
                mapping = {unique[0]: 0, unique[1]: 1}
                return np.array([mapping[v] for v in vals])
            return vals

    raise ValueError("Could not find a label column. Specify label_col=...")


def extract_batch_labels(df, batch_col=None):
    """Extract batch/site labels from a dataframe."""
    if batch_col is not None:
        return df[batch_col].values

    candidates = ["batch", "plate", "site", "cohort", "center", "Batch", "Plate", "Site"]
    for col in candidates:
        if col in df.columns:
            return df[col].values

    return None


def log_transform(X, pseudocount=1.0):
    """Log-transform feature matrix (common for metabolomics intensity data)."""
    return np.log2(X + pseudocount)


def preprocess_cohorts(df1, df2, label_col=None, batch_col=None,
                       log_transform_data=True, feature_cols=None):
    """Full preprocessing pipeline for two cohort dataframes.

    Returns dict with aligned matrices, labels, and metadata.
    """
    print(f"[{datetime.now():%H:%M:%S}] Preprocessing cohorts...")

    X1, X2, features = align_features(df1, df2, feature_cols)
    y1 = extract_labels(df1, label_col)
    y2 = extract_labels(df2, label_col)
    batch1 = extract_batch_labels(df1, batch_col)
    batch2 = extract_batch_labels(df2, batch_col)

    n_missing_1 = np.isnan(X1).sum()
    n_missing_2 = np.isnan(X2).sum()
    if n_missing_1 > 0 or n_missing_2 > 0:
        print(f"  Warning: {n_missing_1} missing values in C1, {n_missing_2} in C2")
        col_means_1 = np.nanmean(X1, axis=0)
        col_means_2 = np.nanmean(X2, axis=0)
        for j in range(X1.shape[1]):
            mask1 = np.isnan(X1[:, j])
            mask2 = np.isnan(X2[:, j])
            X1[mask1, j] = col_means_1[j]
            X2[mask2, j] = col_means_2[j]

    if log_transform_data:
        if np.any(X1 < 0) or np.any(X2 < 0):
            print("  Skipping log-transform (negative values present)")
        else:
            X1 = log_transform(X1)
            X2 = log_transform(X2)
            print("  Applied log2 transform")

    print(f"  C1: {X1.shape[0]} samples x {X1.shape[1]} features, "
          f"{int(y1.sum())}/{len(y1)} positive")
    print(f"  C2: {X2.shape[0]} samples x {X2.shape[1]} features, "
          f"{int(y2.sum())}/{len(y2)} positive")

    return {
        "X1": X1, "X2": X2,
        "y1": y1, "y2": y2,
        "batch1": batch1, "batch2": batch2,
        "features": features,
        "n_features": len(features),
    }

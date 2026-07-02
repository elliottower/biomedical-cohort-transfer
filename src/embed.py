"""Embedding backends for metabolomics feature matrices.

Three backends (config-selectable):
1. raw_features — standardized peak-intensity matrix (always available)
2. classical — PCA or UMAP dimensionality reduction
3. lsm — LSM-MS2 embeddings (gated, requires Pyxis API or self-supervised)
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def embed_raw_features(X):
    """Standardize the feature matrix. No dimensionality reduction."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def embed_classical(X, method="pca", n_components=50):
    """PCA or UMAP embedding of the feature matrix."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])

    if method == "pca":
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X_scaled)
    elif method == "umap":
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("UMAP not installed. Run: pip install umap-learn")
        reducer = UMAP(n_components=n_components)
        return reducer.fit_transform(X_scaled)
    else:
        raise ValueError(f"Unknown method: {method}")


def embed_lsm(X, spectra=None):
    """LSM-MS2 embeddings (stub — not yet implemented).

    Requires either Pyxis free-tier API access or self-supervised
    pretraining on pooled unlabeled spectra.
    """
    raise NotImplementedError(
        "LSM embedding requires Pyxis API access or self-supervised pretraining. "
        "Set use_lsm: false in config.yaml."
    )


def embed(X, backend="raw_features", **kwargs):
    """Dispatch to the appropriate embedding backend."""
    if backend == "raw_features":
        return embed_raw_features(X)
    elif backend == "classical":
        return embed_classical(X, **kwargs)
    elif backend == "lsm":
        return embed_lsm(X, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

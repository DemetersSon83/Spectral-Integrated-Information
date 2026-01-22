from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer


@dataclass
class PhiResult:
    phi_raw: float
    phi_norm: float
    partition: np.ndarray  # boolean mask for group A (True) vs B (False)
    fiedler_vector: np.ndarray


def discretize_timeseries(X: np.ndarray, n_bins: int = 8, *, strategy: str = "uniform") -> np.ndarray:
    """
    Discretize each node's time series into bins for MI estimation.

    Args:
        X: shape (n_nodes, T)
    Returns:
        X_disc: shape (n_nodes, T) integer bins
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_nodes, T).")
    est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy, subsample=None)
    # KBins expects (n_samples, n_features) => use time as samples, nodes as features.
    X_disc = est.fit_transform(X.T).T
    return X_disc.astype(int)


def mutual_info_matrix(X: np.ndarray, n_bins: int = 8, *, strategy: str = "uniform") -> np.ndarray:
    """
    Pairwise mutual information matrix for discretized signals.

    Following the manuscript workflow: discretize each time series into b bins,
    then estimate MI via empirical counts. fileciteturn0file0L144-L165
    """
    X_disc = discretize_timeseries(X, n_bins=n_bins, strategy=strategy)
    n = X_disc.shape[0]
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            mi = mutual_info_score(X_disc[i], X_disc[j])
            M[i, j] = M[j, i] = float(mi)
    np.fill_diagonal(M, 0.0)
    return M


def spectral_phi(MI: np.ndarray) -> PhiResult:
    """
    Compute Φ_spectral using the Fiedler bipartition of the normalized Laplacian.

    Definition follows Eq. (Φ_spectral = sum_{i in A, j in B} M_ij).
    """
    if MI.ndim != 2 or MI.shape[0] != MI.shape[1]:
        raise ValueError("MI must be a square matrix.")
    if MI.shape[0] < 2:
        return PhiResult(phi_raw=0.0, phi_norm=0.0, partition=np.array([True]), fiedler_vector=np.array([0.0]))

    # Normalized Laplacian; scipy handles degree zeros gracefully.
    L = laplacian(MI, normed=True)
    eigvals, eigvecs = eigh(L)
    fiedler = eigvecs[:, 1]
    part = fiedler >= 0

    # Raw cut weight over i<j
    n = MI.shape[0]
    phi_raw = 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            w = float(MI[i, j])
            total += w
            if part[i] != part[j]:
                phi_raw += w

    phi_norm = (phi_raw / total) if total > 0 else 0.0
    return PhiResult(phi_raw=phi_raw, phi_norm=phi_norm, partition=part, fiedler_vector=fiedler)


def phi_timeseries(
    X: np.ndarray,
    *,
    window: int = 50,
    step: int = 5,
    n_bins: int = 8,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[PhiResult]]:
    """
    Sliding-window Φ_spectral(t).

    Returns:
        t_center: int indices (center of each window)
        phi_raw: float array
        phi_norm: float array
        results: list[PhiResult]
    """
    if window <= 1:
        raise ValueError("window must be >= 2.")
    if step < 1:
        raise ValueError("step must be >= 1.")
    n, T = X.shape
    if T < window:
        raise ValueError(f"Time series length T={T} is shorter than window={window}.")

    centers = []
    raws = []
    norms = []
    results: List[PhiResult] = []

    for start in range(0, T - window + 1, step):
        end = start + window
        Xw = X[:, start:end]
        MI = mutual_info_matrix(Xw, n_bins=n_bins, strategy=strategy)
        r = spectral_phi(MI)
        centers.append(start + window // 2)
        raws.append(r.phi_raw)
        norms.append(r.phi_norm)
        results.append(r)

    return np.asarray(centers, dtype=int), np.asarray(raws, dtype=float), np.asarray(norms, dtype=float), results

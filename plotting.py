from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def ensure_outdir(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def plot_phi(t: np.ndarray, phi: np.ndarray, *, title: str, outpath: Path) -> None:
    plt.figure()
    plt.plot(t, phi)
    plt.xlabel("time (window center index)")
    plt.ylabel("Φ_spectral")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_phi_overlay(t: np.ndarray, phis: Sequence[np.ndarray], labels: Sequence[str], *, title: str, outpath: Path) -> None:
    plt.figure()
    for y, lab in zip(phis, labels):
        plt.plot(t, y, label=lab)
    plt.xlabel("time (window center index)")
    plt.ylabel("Φ_spectral")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_mi_matrix(MI: np.ndarray, *, title: str, outpath: Path) -> None:
    plt.figure()
    plt.imshow(MI, aspect="equal")
    plt.title(title)
    plt.colorbar(label="I(x_i; x_j)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_traces(X: np.ndarray, *, title: str, outpath: Path, max_traces: int = 50) -> None:
    """
    Heatmap of amplitudes + overlay of traces.
    """
    n, T = X.shape

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(X, aspect="auto")
    ax0.set_title(f"{title} — trace amplitude")
    ax0.set_ylabel("node")

    ax1 = fig.add_subplot(gs[1, 0])
    take = min(n, max_traces)
    for i in range(take):
        ax1.plot(np.arange(T), X[i], linewidth=0.8, alpha=0.8)
    ax1.set_title(f"{title} — agent traces (first {take})")
    ax1.set_xlabel("time")
    ax1.set_ylabel("x")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_phi_vs_window(windows: Sequence[int], means: Sequence[float], stds: Optional[Sequence[float]], *, title: str, outpath: Path) -> None:
    plt.figure()
    plt.plot(windows, means, marker=".")
    if stds is not None:
        plt.errorbar(windows, means, yerr=stds, fmt="none", capsize=3)
    plt.xlabel("window size")
    plt.ylabel("mean Φ_spectral")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

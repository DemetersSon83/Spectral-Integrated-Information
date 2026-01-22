from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import networkx as nx


def random_oscillators(n: int, T: int, *, fmin: float = 0.01, fmax: float = 0.05, seed: Optional[int] = None) -> np.ndarray:
    """
    Random (uncoupled) oscillators: x_i(t) = sin(2π f_i t + φ_i).
    """
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(fmin, fmax, size=n)
    phases = rng.uniform(0.0, 2*np.pi, size=n)
    t = np.arange(T)
    X = np.sin(2*np.pi*freqs[:, None]*t[None, :] + phases[:, None])
    return X.astype(float)


def synchronized_oscillators(n: int, T: int, *, f: float = 0.02, phase: float = 0.0) -> np.ndarray:
    """
    Perfect synchrony: x_i(t) = sin(2π f t) for all i.
    """
    t = np.arange(T)
    sig = np.sin(2*np.pi*f*t + phase)
    return np.tile(sig[None, :], (n, 1)).astype(float)


def transitional_kuramoto(
    n: int,
    T: int,
    *,
    wmin: float = 0.01,
    wmax: float = 0.05,
    k_max: float = 2.0,
    k_mid: float = 0.6,
    k_steep: float = 12.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Transitional oscillators that gradually synchronize.

    Discrete-time Kuramoto-ish update (as in manuscript description):
        θ_{t+1,i} = θ_{t,i} + ω_i + K(t) * sin(θ̄_t - θ_{t,i})
        x_{t,i} = sin(θ_{t,i})

    K(t) is a sigmoid ramp from ~0 to k_max.
    """
    rng = np.random.default_rng(seed)
    omega = rng.uniform(wmin, wmax, size=n) * 2*np.pi  # convert to rad/step-ish
    theta = rng.uniform(0.0, 2*np.pi, size=n)

    X = np.zeros((n, T), dtype=float)
    for t in range(T):
        # sigmoid in normalized time u∈[0,1]
        u = t / max(1, (T - 1))
        Kt = k_max / (1.0 + np.exp(-k_steep*(u - k_mid)))
        mean_phase = np.angle(np.mean(np.exp(1j*theta)))
        theta = theta + omega + Kt * np.sin(mean_phase - theta)
        X[:, t] = np.sin(theta)

    return X


def ctln(
    n: int,
    T: int,
    *,
    p_edge: float = 0.15,
    theta_drive: float = 1.0,
    eps: float = 0.25,
    delta: float = 0.5,
    dt: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, nx.DiGraph]:
    """
    Combinatorial Threshold-Linear Network (CTLN) simulation.

    Dynamics:
        dx_i/dt = -x_i + [ sum_j W_ij x_j + θ ]_+
    where
        W_ii = 0
        W_ij = -1 + eps if j -> i is an edge
        W_ij = -1 - delta otherwise

    We generate a random directed graph with edge probability p_edge.
    Returns:
        X : (n, T) time series
        G : underlying directed graph
    """
    rng = np.random.default_rng(seed)
    # Random directed graph (no self-loops)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            if rng.random() < p_edge:
                G.add_edge(j, i)

    W = np.full((n, n), -1.0 - delta, dtype=float)
    np.fill_diagonal(W, 0.0)
    for (j, i) in G.edges():
        W[i, j] = -1.0 + eps  # note: j->i corresponds to influence of x_j on x_i

    x = rng.random(n) * 0.1
    X = np.zeros((n, T), dtype=float)
    for t in range(T):
        inp = W @ x + theta_drive
        inp = np.maximum(inp, 0.0)
        dx = -x + inp
        x = x + dt * dx
        X[:, t] = x

    return X, G

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd


def _is_numeric_series(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s.dropna().iloc[:50], errors="raise")
        return True
    except Exception:
        return False


def load_timeseries(
    path: Union[str, Path],
    *,
    fmt: str = "auto",
    drop_first_col: bool = False,
    long_time_col: str = "t",
    long_node_col: str = "node",
    long_value_col: str = "value",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[list]]:
    """
    Load multivariate time-series data as X with shape (n_nodes, T).

    Returns:
        X : np.ndarray (n_nodes, T)
        t : np.ndarray or None
        node_labels : list[str] or None
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if fmt == "auto":
        if suffix in [".csv", ".tsv"]:
            fmt = "csv"
        elif suffix in [".json"]:
            fmt = "json"
        elif suffix in [".npy"]:
            fmt = "npy"
        else:
            raise ValueError(f"Cannot infer format from extension {suffix!r}. Use --format.")

    if fmt == "npy":
        X = np.load(path)
        if X.ndim != 2:
            raise ValueError("Expected .npy to contain a 2D array shaped (n_nodes, T).")
        return X.astype(float), None, None

    if fmt == "csv":
        sep = "," if suffix == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)

        if drop_first_col and df.shape[1] >= 2:
            # Treat first column as time / index
            t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
            df = df.iloc[:, 1:]
        else:
            # Heuristic: if first col looks like time, keep it as t and drop
            t = None
            if df.shape[1] >= 2 and _is_numeric_series(df.iloc[:, 0]) and not _is_numeric_series(df.iloc[:, 1]):
                # second column not numeric => likely not wide
                t = None
            elif df.shape[1] >= 2 and _is_numeric_series(df.iloc[:, 0]) and all(_is_numeric_series(df[c]) for c in df.columns[1:]):
                # first col numeric and rest numeric => probably time in first col
                t = df.iloc[:, 0].to_numpy()
                df = df.iloc[:, 1:]

        # Keep only numeric columns
        numeric_cols = [c for c in df.columns if _is_numeric_series(df[c])]
        if not numeric_cols:
            raise ValueError("CSV parsing produced no numeric columns. If you have long-form data, use --format long-json/csv and convert first.")
        df = df[numeric_cols]
        node_labels = [str(c) for c in df.columns]
        X = df.to_numpy().T
        return X.astype(float), t, node_labels

    if fmt == "json":
        raw = json.loads(path.read_text(encoding="utf-8"))

        # dict of arrays
        if isinstance(raw, dict):
            # try to pull out a time vector if present
            t = None
            if "t" in raw and isinstance(raw["t"], list):
                t = np.asarray(raw["t"], dtype=float)
                raw = {k: v for k, v in raw.items() if k != "t"}

            keys = list(raw.keys())
            if not keys:
                raise ValueError("Empty JSON dict.")
            series = []
            for k in keys:
                if not isinstance(raw[k], list):
                    raise ValueError(f"JSON dict values must be lists. Key={k!r} has type {type(raw[k])}.")
                series.append(np.asarray(raw[k], dtype=float))
            T = min(len(s) for s in series)
            series = [s[:T] for s in series]
            X = np.vstack(series)
            return X.astype(float), t[:T] if t is not None else None, keys

        # list of records (long form)
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
            required = {long_time_col, long_node_col, long_value_col}
            if not required.issubset(df.columns):
                raise ValueError(
                    "Long-form JSON must contain keys "
                    f"{sorted(required)}. Found columns={list(df.columns)}"
                )
            # pivot to wide
            df[long_time_col] = pd.to_numeric(df[long_time_col], errors="coerce")
            df[long_value_col] = pd.to_numeric(df[long_value_col], errors="coerce")
            df = df.dropna(subset=[long_time_col, long_node_col, long_value_col])
            pivot = df.pivot_table(index=long_time_col, columns=long_node_col, values=long_value_col, aggfunc="mean").sort_index()
            t = pivot.index.to_numpy(dtype=float)
            node_labels = [str(c) for c in pivot.columns]
            X = pivot.to_numpy().T
            return X.astype(float), t, node_labels

        raise ValueError(f"Unsupported JSON structure type: {type(raw)}")

    raise ValueError(f"Unknown fmt={fmt!r}")


def save_timeseries_csv(path: Union[str, Path], X: np.ndarray, t: Optional[np.ndarray] = None, node_labels: Optional[list] = None) -> None:
    """Save X (n_nodes, T) to CSV in wide format (rows=time, cols=nodes)."""
    path = Path(path)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_nodes, T).")
    n, T = X.shape
    if node_labels is None:
        node_labels = [f"node{i}" for i in range(n)]
    df = pd.DataFrame(X.T, columns=node_labels)
    if t is not None:
        df.insert(0, "t", t[:T])
    df.to_csv(path, index=False)

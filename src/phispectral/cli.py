from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .io import load_timeseries, save_timeseries_csv
from .metrics import phi_timeseries, mutual_info_matrix
from .plotting import ensure_outdir, plot_phi, plot_mi_matrix, plot_traces, plot_phi_overlay, plot_phi_vs_window
from . import simulate as sim


def _parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def cmd_compute(args: argparse.Namespace) -> int:
    outdir = ensure_outdir(Path(args.outdir))

    X, t, labels = load_timeseries(
        args.input,
        fmt=args.format,
        drop_first_col=args.drop_first_col,
        long_time_col=args.long_time_col,
        long_node_col=args.long_node_col,
        long_value_col=args.long_value_col,
    )
    n, T = X.shape

    centers, phi_raw, phi_norm, results = phi_timeseries(
        X,
        window=args.window,
        step=args.step,
        n_bins=args.bins,
        strategy=args.binning,
    )

    phi = phi_norm if args.normalize else phi_raw
    title = args.title or f"Φ_spectral ({'norm' if args.normalize else 'raw'})"

    # Save CSV
    df = pd.DataFrame({
        "t_center": centers,
        "phi_raw": phi_raw,
        "phi_norm": phi_norm,
    })
    df.to_csv(outdir / "phi_timeseries.csv", index=False)

    # Plots
    plot_phi(centers, phi, title=title, outpath=outdir / "phi_timeseries.png")
    plot_traces(X, title=args.trace_title or "Input data", outpath=outdir / "traces.png")

    # Final window MI
    last_start = (T - args.window) if (T - args.window) >= 0 else 0
    MI_final = mutual_info_matrix(X[:, last_start:last_start + args.window], n_bins=args.bins, strategy=args.binning)
    plot_mi_matrix(MI_final, title="Mutual information (final window)", outpath=outdir / "mi_final.png")

    if args.save_mi_final:
        np.save(outdir / "mi_final.npy", MI_final)

    print(f"Wrote outputs to: {outdir}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    outdir = ensure_outdir(Path(args.outdir))
    X, t, labels = load_timeseries(
        args.input,
        fmt=args.format,
        drop_first_col=args.drop_first_col,
        long_time_col=args.long_time_col,
        long_node_col=args.long_node_col,
        long_value_col=args.long_value_col,
    )
    windows = _parse_int_list(args.windows)
    windows = sorted(set(windows))

    all_means = []
    all_stds = []
    series = []
    centers_ref = None

    for w in windows:
        wdir = ensure_outdir(outdir / f"window_{w}")
        centers, phi_raw, phi_norm, _ = phi_timeseries(X, window=w, step=args.step, n_bins=args.bins, strategy=args.binning)
        phi = phi_norm if args.normalize else phi_raw
        # store
        pd.DataFrame({"t_center": centers, "phi_raw": phi_raw, "phi_norm": phi_norm}).to_csv(wdir / "phi_timeseries.csv", index=False)
        plot_phi(centers, phi, title=f"Φ_spectral (window={w})", outpath=wdir / "phi_timeseries.png")
        all_means.append(float(np.mean(phi)))
        all_stds.append(float(np.std(phi)))
        series.append(phi)
        if centers_ref is None:
            centers_ref = centers

    # Overlay plot (only if centers align)
    if args.overlay and centers_ref is not None and all((len(s)==len(series[0])) for s in series):
        labels = [f"w={w}" for w in windows]
        plot_phi_overlay(centers_ref, series, labels, title="Φ_spectral across window sizes", outpath=outdir / "phi_overlay.png")

    plot_phi_vs_window(windows, all_means, all_stds if args.errorbars else None, title="Mean Φ_spectral vs window size", outpath=outdir / "phi_vs_window.png")
    pd.DataFrame({"window": windows, "mean_phi": all_means, "std_phi": all_stds}).to_csv(outdir / "phi_vs_window.csv", index=False)

    print(f"Wrote sweep outputs to: {outdir}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    model = args.model.lower()

    if model == "random":
        X = sim.random_oscillators(args.n, args.T, fmin=args.fmin, fmax=args.fmax, seed=args.seed)
    elif model == "synchronized":
        X = sim.synchronized_oscillators(args.n, args.T, f=args.f)
    elif model == "transitional":
        X = sim.transitional_kuramoto(
            args.n, args.T,
            wmin=args.wmin, wmax=args.wmax,
            k_max=args.k_max, k_mid=args.k_mid, k_steep=args.k_steep,
            seed=args.seed
        )
    elif model == "ctln":
        X, G = sim.ctln(
            args.n, args.T,
            p_edge=args.p_edge,
            theta_drive=args.theta_drive,
            eps=args.eps,
            delta=args.delta,
            dt=args.dt,
            seed=args.seed
        )
        if args.save_graph:
            import networkx as nx
            nx.write_gml(G, outpath.with_suffix(".gml"))
    else:
        raise SystemExit(f"Unknown model: {args.model}. Choose from random|transitional|synchronized|ctln.")

    if outpath.suffix.lower() == ".npy":
        np.save(outpath, X)
    else:
        save_timeseries_csv(outpath, X)

    print(f"Saved {model} data to {outpath} (shape={X.shape})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phispectral", description="Compute Φ_spectral on multivariate time-series.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # shared input args
    def add_input(sp):
        sp.add_argument("--input", required=True, help="Path to input time-series (.csv/.json/.npy).")
        sp.add_argument("--format", default="auto", help="auto|csv|json|npy")
        sp.add_argument("--drop-first-col", action="store_true", help="If CSV includes a time column first, drop it.")
        sp.add_argument("--long-time-col", default="t", help="For long-form JSON records: time column key.")
        sp.add_argument("--long-node-col", default="node", help="For long-form JSON records: node column key.")
        sp.add_argument("--long-value-col", default="value", help="For long-form JSON records: value column key.")

    def add_phi_params(sp):
        sp.add_argument("--window", type=int, default=50, help="Sliding window size (time steps).")
        sp.add_argument("--step", type=int, default=5, help="Step between windows.")
        sp.add_argument("--bins", type=int, default=8, help="Bins for discretization.")
        sp.add_argument("--binning", default="uniform", help="KBinsDiscretizer strategy (uniform|quantile|kmeans).")
        sp.add_argument("--normalize", action="store_true", help="Use normalized Φ (phi_raw / total_MI).")
        sp.add_argument("--title", default=None, help="Plot title for Φ(t).")
        sp.add_argument("--trace-title", default=None, help="Plot title for traces.")
        sp.add_argument("--outdir", default="phispectral_outputs", help="Output directory.")
        sp.add_argument("--save-mi-final", action="store_true", help="Save final MI matrix as .npy too.")

    sp_compute = sub.add_parser("compute", help="Compute Φ_spectral(t) for an input time-series.")
    add_input(sp_compute)
    add_phi_params(sp_compute)
    sp_compute.set_defaults(func=cmd_compute)

    sp_sweep = sub.add_parser("sweep", help="Run Φ_spectral(t) across multiple window sizes.")
    add_input(sp_sweep)
    sp_sweep.add_argument("--windows", required=True, help="Comma-separated window sizes, e.g. 10,20,50,100")
    sp_sweep.add_argument("--step", type=int, default=5)
    sp_sweep.add_argument("--bins", type=int, default=8)
    sp_sweep.add_argument("--binning", default="uniform")
    sp_sweep.add_argument("--normalize", action="store_true")
    sp_sweep.add_argument("--overlay", action="store_true", help="Also write an overlay plot across windows (if alignable).")
    sp_sweep.add_argument("--errorbars", action="store_true", help="Add std-dev error bars to mean-vs-window plot.")
    sp_sweep.add_argument("--outdir", default="phispectral_sweep_outputs")
    sp_sweep.set_defaults(func=cmd_sweep)

    sp_sim = sub.add_parser("simulate", help="Generate toy-model data used in the manuscript.")
    sp_sim.add_argument("--model", required=True, help="random|transitional|synchronized|ctln")
    sp_sim.add_argument("--n", type=int, default=50)
    sp_sim.add_argument("--T", type=int, default=150)
    sp_sim.add_argument("--seed", type=int, default=None)
    sp_sim.add_argument("--out", required=True, help="Output file (.csv or .npy).")

    # random params
    sp_sim.add_argument("--fmin", type=float, default=0.01)
    sp_sim.add_argument("--fmax", type=float, default=0.05)
    # synchronized params
    sp_sim.add_argument("--f", type=float, default=0.02)
    # transitional params
    sp_sim.add_argument("--wmin", type=float, default=0.01)
    sp_sim.add_argument("--wmax", type=float, default=0.05)
    sp_sim.add_argument("--k-max", type=float, default=2.0)
    sp_sim.add_argument("--k-mid", type=float, default=0.6)
    sp_sim.add_argument("--k-steep", type=float, default=12.0)
    # ctln params
    sp_sim.add_argument("--p-edge", type=float, default=0.15)
    sp_sim.add_argument("--theta-drive", type=float, default=1.0)
    sp_sim.add_argument("--eps", type=float, default=0.25)
    sp_sim.add_argument("--delta", type=float, default=0.5)
    sp_sim.add_argument("--dt", type=float, default=0.1)
    sp_sim.add_argument("--save-graph", action="store_true", help="If CTLN, also save the random graph as .gml")

    sp_sim.set_defaults(func=cmd_simulate)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

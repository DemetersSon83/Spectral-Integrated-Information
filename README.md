# Spectral Integrated Information

(C) 2026 Mark M. Bailey, PhD

A small Python package + CLI to compute **Φ_spectral** (spectral integrated information) on multivariate time-series,
as defined in [*When Wholes Resist Decomposition: A Spectral Measure of Epistemic Emergence*](https://philarchive.org/rec/BAIWWR).

## Install

```bash
pip install -e .
```

## Input formats

The CLI expects an `n_nodes × T` matrix `X` (nodes/agents by time).

Supported on-disk formats:

- **CSV**: "wide" by default (columns are nodes; rows are time). If your first column is time, pass `--drop-first-col`.
- **JSON**:
  - dict of lists: `{ "node1": [...], "node2": [...] }`
  - list of records: `[{"t":0,"node": "a","value": 0.1}, ...]` (use `--format long`)

You can always bypass parsing by saving a NumPy `.npy` file and using `--input my.npy`.

## Common tasks

Compute Φ_spectral(t) with a sliding window:

```bash
phispectral compute --input data.csv --window 50 --step 5 --bins 8 --outdir outputs
```

Sweep across multiple window sizes and generate plots:

```bash
phispectral sweep --input data.csv --windows 10,20,50,100 --step 5 --bins 8 --outdir outputs
```

Generate toy-model data (random/transitional/synchronized/ctln), save it, and compute Φ:

```bash
phispectral simulate --model transitional --n 50 --T 150 --seed 7 --out data.csv
phispectral compute --input data.csv --window 10 --step 1 --bins 3 --outdir outputs
```

## Output

`compute` writes:

- `phi_timeseries.csv` : time index, phi_raw, phi_norm
- `phi_timeseries.png` : Φ(t)
- `mi_final.png` : mutual-information matrix for the last window
- `traces.png` : trace heatmap + traces

`sweep` writes the above for each window and additionally:

- `phi_vs_window.png` : mean Φ vs window size (plus optional spread)

## Citation

```bibtex
@misc{BaileySchneider2025,
  author       = {Mark M. Bailey and Susan L. Schneider},
  title        = {When Wholes Resist Decomposition: A Spectral Measure of Epistemic Emergence},
  note         = {PhilPapers preprint},
  year         = {2025},
  url          = {https://philpapers.org/rec/BAIWWR},
}
```

# Frame-calibrated Sterimol reconstruction

This repository reproduces frame-calibrated Sterimol reconstruction (FCSR) on
the 1,552-ligand Kraken conformer-ensemble benchmark. FCSR reconstructs a
phosphorus-centred molecular frame, evaluates analytic van der Waals extrema
for each conformer, averages them with the released population weights, and
fits a one-dimensional calibration to the reference convention.

The compact deposit contains source code, five-seed numerical results and the
quantitative figure programs. The raw Kraken archive, generated feature cache
and rendered figures are intentionally excluded.

## Repository layout

| Path | Contents |
|---|---|
| `fcsr/` | Kraken loading, molecular features, splits and shared utilities |
| `scripts/run_sterimol_axis_formula.py` | Analytic conformer features and calibration-only candidate selection |
| `scripts/run_fcsr_analysis.py` | Fixed and nested FCSR, decomposition, sensitivity and stratified analyses |
| `scripts/run_frame_controls.py` | Bond-frame, principal-axis and coordinate-frame controls |
| `scripts/run_supporting_analyses.py` | Conformer-budget, missing-weight, scaffold and conformer-count analyses |
| `scripts/validate_results.py` | Numerical regression checks against the deposited tables |
| `reference_results/` | Aggregate and per-seed CSV results used in the paper |
| `figures/` | Quantitative figure code and required source data |

## Environment

The numerical experiments were run with the pinned Python 3.10 environment:

```bash
conda env create -f environment.yml
conda activate fcsr
python -m pip install -e . --no-deps
pytest -q
```

For a faster installation in mainland China, a local conda mirror may be
configured before creating the environment. Package versions should remain
unchanged for numerical comparison.

## Data

Download the Kraken archive released with the MARCEL benchmark:

```bash
python scripts/download_data.py
```

This writes `data/Kraken.zip` (ignored by Git). The downloader uses the public
archive identifier distributed by MARCEL. The original dataset remains subject
to its source terms; the MIT license in this repository covers only the code.

Primary data and benchmark sources:

- [Kraken ligand platform](https://doi.org/10.1021/jacs.1c09718)
- [MARCEL benchmark and data release](https://github.com/SXKDZ/MARCEL)
- [MolMix source for most reported comparator values](https://arxiv.org/abs/2410.07981)
- [FACET source for the FACET-GemNet value](https://openreview.net/forum?id=cpwbXHvd2h)

## Reproduce the experiments

Run commands from the repository root. Feature extraction creates a local
cache from 21,287 conformers:

```bash
python scripts/run_sterimol_axis_formula.py --features-only
python scripts/run_sterimol_axis_formula.py --features-only --axis-mode all --radii 4.0 --cache data/frame_rows.joblib
```

The full paper analyses use seeds 0--4 and a 70/10/20
fit/calibration/test partition:

```bash
python scripts/run_fcsr_analysis.py
python scripts/run_frame_controls.py
python scripts/run_supporting_analyses.py
python scripts/validate_results.py --strict --include-perturbations
```

The validator uses a 1e-4 absolute and relative tolerance to accommodate
platform-level solver differences. With the pinned environment, stricter
comparison is available through `--rtol 1e-7 --atol 1e-9`.

`run_fcsr_analysis.py` includes geometry perturbations and the diagnostic frame
audit. A shorter numerical check can omit those two computations:

```bash
python scripts/run_fcsr_analysis.py --skip-axis-diagnostic --skip-geometry-noise
python scripts/validate_results.py
```

Candidate and calibrator selection in the nested analysis uses only the
calibration partition. Test labels are evaluated once after selection. The
fixed analysis uses the predeclared target-specific formulas documented in the
script. Conformer target labels appear only in the error-decomposition
diagnostic and are not prediction inputs.

## Reproduce the quantitative figures

The final plotting programs read `reference_results/` and write PDF, SVG and
600 dpi PNG files into `figures/`:

```bash
python figures/prepare_applicability_domain.py
python figures/plot_figures_3_4_s6.py
python figures/plot_remaining_data_figures.py
python figures/plot_reorganized_figures.py
```

The publication typography uses Times New Roman. Install that font before
running the plotting programs, or change `FONT_FAMILY` in `figures/palette.py`
for a local preview.

## Result provenance

FCSR aggregate and per-seed values in `reference_results/selection/`,
`reference_results/analysis/` and `reference_results/supporting/` are outputs
of this implementation. `reference_results/supporting/external_baselines.csv`
contains values transcribed from the cited MolMix and FACET tables; its
`provenance` column marks them as source-reported rather than rerun here.

The principal fixed five-seed MAEs are 0.0287, 0.0530, 0.0938 and 0.0567
angstrom for B5, L, buried B5 and buried L, respectively. Full-precision values,
standard deviations and companion metrics are retained in the CSV files.

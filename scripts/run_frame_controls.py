#!/usr/bin/env python
"""Evaluate coordinate-frame controls under the five-seed Kraken protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fcsr.kraken import load_kraken  # noqa: E402
from run_sterimol_axis_formula import (  # noqa: E402
    _aggregate_formula_predictions,
    _calibrated_predictions,
    _metrics,
    _random_split,
)


FRAME_SPECS = {
    "sterimol_B5": ("Bmax_all", "huber"),
    "sterimol_L": ("Lpos_all", "huber"),
}
AXES = ["sum_pc", "pc0", "pc1", "pc2", "pca0", "pca1", "pca2", "lone_pair", "global_x"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kraken-zip", type=Path, default=ROOT / "data/Kraken.zip")
    parser.add_argument("--formula-cache", type=Path, default=ROOT / "data/frame_rows.joblib")
    parser.add_argument("--out", type=Path, default=ROOT / "results/frame")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    molecules, _ = load_kraken(args.kraken_zip, download=False)
    conformers = joblib.load(args.formula_cache)
    candidates = [f"{axis}_{feature}" for feature, _ in FRAME_SPECS.values() for axis in AXES]
    missing = sorted(set(candidates).difference(conformers.columns))
    if missing:
        raise SystemExit(f"Feature cache lacks frame controls: {missing}")
    aggregated = _aggregate_formula_predictions(conformers, candidates, ["L4_32"], "boltzmann")["L4_32"]
    indexed = molecules.set_index("molecule_id")
    rows: list[dict[str, object]] = []

    for seed in args.seeds:
        split = _random_split(molecules, seed, (0.7, 0.0, 0.1, 0.2))
        split.index = molecules["molecule_id"].astype(str)
        for target, (feature, calibration) in FRAME_SPECS.items():
            y = indexed[target].astype(float)
            for axis in AXES:
                candidate = f"{axis}_{feature}"
                x = aggregated[candidate].reindex(y.index).to_numpy(dtype=float)
                truth = y.to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(truth)
                fit = split.reindex(y.index).isin(["train", "val"]).to_numpy() & valid
                test = split.reindex(y.index).eq("test").to_numpy() & valid
                pred = _calibrated_predictions(calibration, x[fit], truth[fit], x[valid])
                full = np.full_like(truth, np.nan)
                full[np.flatnonzero(valid)] = pred
                score = _metrics(truth[test], full[test])
                rows.append({"target": target, "axis": axis, "candidate": candidate,
                             "calibration": calibration, "seed": seed, **score})

    raw = pd.DataFrame(rows)
    summary = (raw.groupby(["target", "axis", "candidate", "calibration"], sort=False)
               .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                    rmse_mean=("rmse", "mean"), r2_mean=("r2", "mean"),
                    n_seeds=("seed", "nunique"))
               .reset_index())
    raw.to_csv(args.out / "frame_controls_raw.csv", index=False)
    summary.to_csv(args.out / "frame_controls_5seed.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Compare regenerated numerical tables with the deposited reference results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PAIRS = {
    "analysis/error_decomposition.csv": "analysis/error_decomposition.csv",
    "analysis/nested_target_summary.csv": "analysis/nested_target_summary.csv",
    "analysis/radius_sensitivity.csv": "analysis/radius_sensitivity.csv",
    "analysis/calibration_size.csv": "analysis/calibration_size.csv",
    "analysis/chemical_families.csv": "analysis/chemical_families.csv",
    "supporting/conformer_budget_5seed.csv": "supporting/conformer_budget_5seed.csv",
    "supporting/uniform_weights_5seed.csv": "supporting/uniform_weights_5seed.csv",
    "supporting/weight_ablation_5seed.csv": "supporting/weight_ablation_5seed.csv",
    "supporting/scaffold_split_5seed.csv": "supporting/scaffold_split_5seed.csv",
    "supporting/conformer_count_5seed.csv": "supporting/conformer_count_5seed.csv",
    "frame/frame_controls_5seed.csv": "analysis/frame_controls_5seed.csv",
}


def compare(generated: Path, reference: Path, rtol: float, atol: float) -> None:
    got = pd.read_csv(generated)
    expected = pd.read_csv(reference)
    if list(got.columns) != list(expected.columns):
        raise AssertionError(f"column mismatch: {generated}")
    if len(got) != len(expected):
        raise AssertionError(f"row-count mismatch: {generated} ({len(got)} != {len(expected)})")
    text_cols = [col for col in expected if not pd.api.types.is_numeric_dtype(expected[col])]
    sort_cols = [col for col in text_cols if expected[col].nunique(dropna=False) > 1]
    if sort_cols:
        got = got.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        expected = expected.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    for col in text_cols:
        if not got[col].fillna("<NA>").astype(str).equals(expected[col].fillna("<NA>").astype(str)):
            raise AssertionError(f"text mismatch in {generated}: {col}")
    for col in expected.columns.difference(text_cols):
        if not np.allclose(got[col], expected[col], rtol=rtol, atol=atol, equal_nan=True):
            delta = np.nanmax(np.abs(got[col].to_numpy(float) - expected[col].to_numpy(float)))
            raise AssertionError(f"numeric mismatch in {generated}: {col}, max abs delta={delta:g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, default=Path("results"))
    parser.add_argument("--reference", type=Path, default=Path("reference_results"))
    parser.add_argument("--strict", action="store_true", help="Fail when a generated table is absent.")
    parser.add_argument("--include-perturbations", action="store_true", help="Also validate the full geometry-perturbation table.")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    checked = 0
    pairs = dict(PAIRS)
    if args.include_perturbations:
        pairs["analysis/perturbation_5seed.csv"] = "analysis/perturbation_5seed.csv"
    for generated_name, reference_name in pairs.items():
        generated = args.generated / generated_name
        reference = args.reference / reference_name
        if not generated.exists():
            if args.strict:
                raise FileNotFoundError(generated)
            print(f"[skip] {generated}")
            continue
        compare(generated, reference, args.rtol, args.atol)
        checked += 1
        print(f"[pass] {generated_name}")
    print(f"Validated {checked} result tables.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Reproduce budget, population-weight, scaffold and conformer-count analyses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fcsr.kraken import load_kraken  # noqa: E402
from fcsr.splits import scaffold_split  # noqa: E402
from run_sterimol_axis_formula import (  # noqa: E402
    LEVEL_K,
    _aggregate_formula_predictions,
    _calibrated_predictions,
    _metrics,
    _random_split,
)


SPECS = [
    ("B5", "sterimol_B5", "sum_pc_Bmax_all", "huber", 0.1910),
    ("L", "sterimol_L", "sum_pc_Lpos_all", "huber", 0.3050),
    ("BurB5", "sterimol_burB5", "sum_pc_R4.4_touch_Bmax", "huber", 0.1460),
    ("BurL", "sterimol_burL", "sum_pc_R2.7_touch_Lpos", "isotonic", 0.0947),
]
LEVELS = list(LEVEL_K)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results/supporting")
    parser.add_argument("--kraken-zip", type=Path, default=ROOT / "data/Kraken.zip")
    parser.add_argument("--formula-cache", type=Path, default=ROOT / "data/formula_rows.joblib")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def summarize(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    result = frame.groupby(groups, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(str(part) for part in col if part) for col in result.columns.to_flat_index()]
    return result


def evaluate(
    molecules: pd.DataFrame,
    aggregated: dict[str, pd.DataFrame],
    seeds: list[int],
    split_mode: str,
    weight_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = molecules.set_index("molecule_id")
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for seed in seeds:
        if split_mode == "scaffold":
            split = scaffold_split(molecules, seed=seed, fractions=(0.7, 0.0, 0.1, 0.2))
        else:
            split = _random_split(molecules, seed, (0.7, 0.0, 0.1, 0.2))
        split.index = molecules["molecule_id"].astype(str)
        for prop, target, candidate, calibration, external_mae in SPECS:
            y = indexed[target].astype(float)
            for level in LEVELS:
                x = aggregated[level][candidate].reindex(y.index).to_numpy(dtype=float)
                truth = y.to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(truth)
                fit = split.reindex(y.index).isin(["train", "val"]).to_numpy() & valid
                cal = split.reindex(y.index).eq("cal").to_numpy() & valid
                test = split.reindex(y.index).eq("test").to_numpy() & valid
                pred = _calibrated_predictions(calibration, x[fit], truth[fit], x[valid])
                full = np.full_like(truth, np.nan)
                full[np.flatnonzero(valid)] = pred
                cal_score = _metrics(truth[cal], full[cal])
                test_score = _metrics(truth[test], full[test])
                metric_rows.append(
                    {
                        "property": prop, "target": target, "seed": seed,
                        "split_mode": split_mode, "weight_mode": weight_mode,
                        "level": level, "candidate": candidate, "calibration": calibration,
                        "best_literature_mae": external_mae, "cal_mae": cal_score["mae"],
                        **test_score,
                    }
                )
                if level == "L4_32":
                    table = pd.DataFrame(
                        {
                            "property": prop, "seed": seed, "molecule_id": y.index,
                            "true": truth, "pred": full, "split": split.reindex(y.index).to_numpy(),
                        }
                    )
                    prediction_rows.append(table[table["split"].eq("test")])
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    molecules, _ = load_kraken(args.kraken_zip, download=False)
    conformers = joblib.load(args.formula_cache)
    candidates = [spec[2] for spec in SPECS]
    missing = sorted(set(candidates).difference(conformers.columns))
    if missing:
        raise SystemExit(f"Feature cache lacks fixed candidates: {missing}")

    boltzmann = _aggregate_formula_predictions(conformers, candidates, LEVELS, "boltzmann")
    uniform_agg = _aggregate_formula_predictions(conformers, candidates, LEVELS, "uniform")
    random, predictions = evaluate(molecules, boltzmann, args.seeds, "random", "boltzmann")
    uniform, _ = evaluate(molecules, uniform_agg, args.seeds, "random", "uniform")
    scaffold, _ = evaluate(molecules, boltzmann, args.seeds, "scaffold", "boltzmann")

    metrics = ["mae", "rmse", "r2", "cal_mae"]
    groups = ["property", "target", "split_mode", "weight_mode", "level", "candidate",
              "calibration", "best_literature_mae"]
    budget = summarize(random, groups, metrics)
    budget["relative_gain_vs_best_literature_pct"] = (
        (budget["best_literature_mae"] - budget["mae_mean"])
        / budget["best_literature_mae"] * 100
    )
    budget.to_csv(args.out / "conformer_budget_5seed.csv", index=False)

    fixed_uniform = summarize(uniform[uniform["level"].eq("L4_32")], groups, metrics)
    fixed_boltzmann = summarize(random[random["level"].eq("L4_32")], groups, metrics)
    fixed_uniform.to_csv(args.out / "uniform_weights_5seed.csv", index=False)
    weight_delta = fixed_boltzmann.merge(
        fixed_uniform,
        on=["property", "target", "level", "candidate", "calibration", "best_literature_mae"],
        suffixes=("_boltzmann", "_uniform"),
    )
    weight_delta["missing_weight_mae_delta"] = (
        weight_delta["mae_mean_uniform"] - weight_delta["mae_mean_boltzmann"]
    )
    weight_delta.to_csv(args.out / "weight_ablation_5seed.csv", index=False)

    scaffold_summary = summarize(scaffold[scaffold["level"].eq("L4_32")], groups, metrics)
    scaffold_summary["relative_gain_vs_best_literature_pct"] = (
        (scaffold_summary["best_literature_mae"] - scaffold_summary["mae_mean"])
        / scaffold_summary["best_literature_mae"] * 100
    )
    scaffold_summary.to_csv(args.out / "scaffold_split_5seed.csv", index=False)

    predictions = predictions.merge(molecules[["molecule_id", "n_source_confs"]], on="molecule_id", how="left")
    predictions["conformer_count_bucket"] = pd.qcut(
        predictions["n_source_confs"], q=4,
        labels=["Q1_few_confs", "Q2", "Q3", "Q4_many_confs"], duplicates="drop",
    )
    long_rows = []
    for (prop, seed, bucket), group in predictions.groupby(
        ["property", "seed", "conformer_count_bucket"], observed=True
    ):
        error = np.abs(group["pred"].to_numpy(float) - group["true"].to_numpy(float))
        long_rows.append(
            {"property": prop, "seed": seed, "conformer_count_bucket": str(bucket),
             "n_test": len(group), "n_source_confs_mean": group["n_source_confs"].mean(),
             "mae": error.mean()}
        )
    count_summary = summarize(
        pd.DataFrame(long_rows), ["property", "conformer_count_bucket"],
        ["mae", "n_test", "n_source_confs_mean"],
    )
    count_summary.to_csv(args.out / "conformer_count_5seed.csv", index=False)
    print(f"Wrote supporting results to {args.out}")


if __name__ == "__main__":
    main()

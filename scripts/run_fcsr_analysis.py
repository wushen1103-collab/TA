#!/usr/bin/env python
"""Reproduce FCSR Kraken analyses from a released feature cache.

This script fills the experiments requested after the main Kraken result:

* error decomposition for oracle conformer labels, raw formula, weighting, and
  calibration;
* nested per-seed candidate selection using calibration labels only;
* buried-radius sensitivity;
* calibration-label learning curves;
* Boltzmann temperature and energy-noise robustness;
* descriptor-inferred axis diagnostics and chemical-family stratification.

The exact hidden DFT metal-P reference axis is not available in the released
Kraken pickle because the stored conformer SDF blocks are ligand-only. The axis
diagnostic below is therefore explicitly marked as descriptor-inferred and is
not used for prediction.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fcsr.kraken import KRAKEN_TARGETS, _parse_sdf_mol, _record_parts, load_kraken  # noqa: E402
from run_sterimol_axis_formula import (  # noqa: E402
    _aggregate_formula_predictions,
    _calibrated_predictions,
    _metrics,
    _random_split,
    _safe_unit,
    _sterimol_features_for_axis,
)


SPLIT_FRACTIONS = (0.7, 0.0, 0.1, 0.2)
SEEDS = [0, 1, 2, 3, 4]
LEVEL = "L4_32"
TOP_K = 32
R_KCAL = 0.00198720425864083
T_REF = 298.15

FIXED_TARGETS: dict[str, dict[str, Any]] = {
    "sterimol_B5": {
        "short": "B5",
        "candidate": "sum_pc_Bmax_all",
        "calibration": "huber",
        "external_best_mae": 0.191,
    },
    "sterimol_L": {
        "short": "L",
        "candidate": "sum_pc_Lpos_all",
        "calibration": "huber",
        "external_best_mae": 0.305,
    },
    "sterimol_burB5": {
        "short": "BurB5",
        "candidate": "sum_pc_R4.4_touch_Bmax",
        "calibration": "huber",
        "external_best_mae": 0.146,
    },
    "sterimol_burL": {
        "short": "BurL",
        "candidate": "sum_pc_R2.7_touch_Lpos",
        "calibration": "isotonic",
        "external_best_mae": 0.0947,
    },
}


@dataclass
class ConfRecord:
    molecule_id: str
    smiles: str
    conf_id: str
    rank: int
    weight: float
    coords: np.ndarray
    atomic_nums: np.ndarray
    vdw: np.ndarray
    heavy: np.ndarray
    p_idx: int
    neighbor_indices: list[int]
    true_conf: dict[str, float]
    true_mol: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kraken-zip", type=Path, default=ROOT / "data/Kraken.zip")
    parser.add_argument(
        "--formula-cache",
        type=Path,
        default=ROOT / "data/formula_rows.joblib",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results/analysis")
    parser.add_argument("--axis-sphere", type=int, default=96, help="Directions for diagnostic descriptor-inferred axis.")
    parser.add_argument("--skip-axis-diagnostic", action="store_true")
    parser.add_argument("--skip-geometry-noise", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, out: Path, name: str) -> Path:
    path = out / name
    df.to_csv(path, index=False)
    print(f"[write] {path} ({len(df)} rows)")
    return path


def numeric_summary(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def y_for(mol_df: pd.DataFrame, target: str) -> pd.Series:
    return mol_df.set_index("molecule_id")[target].astype(float)


def mol_order(mol_df: pd.DataFrame) -> list[str]:
    return mol_df["molecule_id"].astype(str).tolist()


def split_masks(mol_df: pd.DataFrame, seed: int) -> dict[str, pd.Series]:
    split = _random_split(mol_df, seed, SPLIT_FRACTIONS)
    split.index = mol_df["molecule_id"].astype(str)
    return {
        "fit": split.isin(["train", "val"]),
        "cal": split.eq("cal"),
        "test": split.eq("test"),
        "split": split,
    }


def fixed_candidate_cols() -> list[str]:
    return [cfg["candidate"] for cfg in FIXED_TARGETS.values()]


def aggregate_level(conf_df: pd.DataFrame, candidate_cols: list[str], weight_mode: str = "boltzmann") -> pd.DataFrame:
    agg = _aggregate_formula_predictions(conf_df, candidate_cols, [LEVEL], weight_mode=weight_mode)[LEVEL]
    return agg.sort_index()


def aggregate_custom_weights(
    conf_df: pd.DataFrame,
    candidate_cols: list[str],
    transform: str,
    parameter: float | None = None,
    rng_seed: int = 0,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(rng_seed)
    for mol_id, group in conf_df.sort_values(["molecule_id", "rank"]).groupby("molecule_id", sort=True):
        top = group.head(TOP_K)
        weights = top["weight"].to_numpy(dtype=float)
        weights = np.clip(weights, 0.0, None)
        if transform == "uniform":
            new_w = np.ones_like(weights, dtype=float)
        elif transform == "temperature":
            temp = float(parameter if parameter is not None else T_REF)
            new_w = np.power(np.clip(weights, 1e-300, None), T_REF / temp)
        elif transform == "energy_noise":
            sigma = float(parameter if parameter is not None else 0.0)
            noise = rng.normal(0.0, sigma, size=len(weights))
            new_w = np.clip(weights, 1e-300, None) * np.exp(-noise / (R_KCAL * T_REF))
        else:
            new_w = weights.copy()
        if not np.isfinite(new_w).all() or float(new_w.sum()) <= 0:
            new_w = np.ones_like(weights, dtype=float)
        new_w = new_w / float(new_w.sum())
        row = {"molecule_id": str(mol_id)}
        for col in candidate_cols:
            vals = top[col].to_numpy(dtype=float)
            mask = np.isfinite(vals) & np.isfinite(new_w)
            row[col] = float(np.sum(vals[mask] * new_w[mask]) / max(float(new_w[mask].sum()), 1e-12)) if mask.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("molecule_id").sort_index()


def apply_calibrator(method: str, x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    fit_mask = np.isfinite(x_fit) & np.isfinite(y_fit)
    eval_mask = np.isfinite(x_eval)
    out = np.full_like(x_eval.astype(float), np.nan, dtype=float)
    if fit_mask.sum() < 2:
        out[eval_mask] = x_eval[eval_mask]
        return out
    try:
        out[eval_mask] = _calibrated_predictions(method, x_fit[fit_mask], y_fit[fit_mask], x_eval[eval_mask])
    except Exception:
        out[eval_mask] = x_eval[eval_mask]
    return out


def evaluate_table(
    agg: pd.DataFrame,
    mol_df: pd.DataFrame,
    target: str,
    candidate: str,
    method: str,
    seed: int,
    fit_n: int | None = None,
    sample_seed: int = 0,
    raw: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    y = y_for(mol_df, target)
    masks = split_masks(mol_df, seed)
    index = y.index.intersection(agg.index)
    fit_ids = masks["fit"][masks["fit"]].index.intersection(index)
    test_ids = masks["test"][masks["test"]].index.intersection(index)
    if fit_n is not None and fit_n < len(fit_ids):
        rng = np.random.default_rng(sample_seed)
        fit_ids = pd.Index(rng.choice(fit_ids.to_numpy(), size=int(fit_n), replace=False))

    x_fit = agg.loc[fit_ids, candidate].to_numpy(dtype=float)
    y_fit = y.loc[fit_ids].to_numpy(dtype=float)
    x_test = agg.loc[test_ids, candidate].to_numpy(dtype=float)
    y_test = y.loc[test_ids].to_numpy(dtype=float)
    pred = x_test if raw or method == "raw" else apply_calibrator(method, x_fit, y_fit, x_test)
    ok = np.isfinite(y_test) & np.isfinite(pred)
    metrics = _metrics(y_test[ok], pred[ok]) if ok.any() else {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    pred_df = pd.DataFrame(
        {
            "molecule_id": test_ids.to_numpy(),
            "seed": seed,
            "target": target,
            "candidate": candidate,
            "calibration": "raw" if raw else method,
            "true": y_test,
            "pred": pred,
            "abs_error": np.abs(pred - y_test),
        }
    )
    return metrics, pred_df


def summarize_evaluations(rows: list[dict[str, Any]], by: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metrics = []
    for keys, group in df.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row["mae_mean"] = float(group["mae"].mean())
        row["mae_std"] = float(group["mae"].std(ddof=1)) if len(group) > 1 else 0.0
        row["rmse_mean"] = float(group["rmse"].mean()) if "rmse" in group else float("nan")
        row["r2_mean"] = float(group["r2"].mean()) if "r2" in group else float("nan")
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        metrics.append(row)
    return pd.DataFrame(metrics)


def load_formula_cache(path: Path) -> pd.DataFrame:
    df = joblib.load(path)
    for col in ["molecule_id", "conf_id"]:
        if col in df:
            df[col] = df[col].astype(str)
    return df


def build_error_decomposition(out: Path, conf_df: pd.DataFrame, mol_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed_cols = fixed_candidate_cols()
    agg_boltz = aggregate_level(conf_df, fixed_cols, "boltzmann")
    agg_uniform = aggregate_level(conf_df, fixed_cols, "uniform")
    rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []

    true_conf_cols = [f"{target}_true_conf" for target in KRAKEN_TARGETS]
    oracle_agg = aggregate_custom_weights(conf_df, true_conf_cols, "boltzmann")
    oracle_agg = oracle_agg.rename(columns={f"{target}_true_conf": target for target in KRAKEN_TARGETS})

    for target, cfg in FIXED_TARGETS.items():
        y = y_for(mol_df, target)
        candidate = cfg["candidate"]
        method = cfg["calibration"]
        for seed in SEEDS:
            masks = split_masks(mol_df, seed)
            test_ids = masks["test"][masks["test"]].index.intersection(oracle_agg.index)
            oracle_pred = oracle_agg.loc[test_ids, target].to_numpy(dtype=float)
            y_test = y.loc[test_ids].to_numpy(dtype=float)
            ok = np.isfinite(oracle_pred) & np.isfinite(y_test)
            met = _metrics(y_test[ok], oracle_pred[ok]) if ok.any() else {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
            rows.append(
                {
                    "target": target,
                    "target_short": cfg["short"],
                    "stage": "oracle_conformer_labels_boltzmann_all_confs",
                    "seed": seed,
                    "candidate": f"{target}_true_conf",
                    "calibration": "none",
                    "weight_mode": "boltzmann",
                    **met,
                    "uses_test_labels_for_prediction": False,
                    "diagnostic_uses_conformer_labels": True,
                }
            )

            for stage, agg, raw, weight_mode in [
                ("raw_formula_boltzmann", agg_boltz, True, "boltzmann"),
                ("calibrated_formula_boltzmann", agg_boltz, False, "boltzmann"),
                ("raw_formula_uniform", agg_uniform, True, "uniform"),
                ("calibrated_formula_uniform", agg_uniform, False, "uniform"),
            ]:
                met, pred_df = evaluate_table(agg, mol_df, target, candidate, method, seed, raw=raw)
                rows.append(
                    {
                        "target": target,
                        "target_short": cfg["short"],
                        "stage": stage,
                        "seed": seed,
                        "candidate": candidate,
                        "calibration": "raw" if raw else method,
                        "weight_mode": weight_mode,
                        **met,
                        "uses_test_labels_for_prediction": False,
                        "diagnostic_uses_conformer_labels": False,
                    }
                )
                if stage == "calibrated_formula_boltzmann":
                    pred_frames.append(pred_df)

    raw = pd.DataFrame(rows)
    summary = summarize_evaluations(
        rows,
        by=["target", "target_short", "stage", "candidate", "calibration", "weight_mode", "uses_test_labels_for_prediction", "diagnostic_uses_conformer_labels"],
    )
    if not summary.empty:
        summary["external_best_mae"] = summary["target"].map({t: cfg["external_best_mae"] for t, cfg in FIXED_TARGETS.items()})
        summary["relative_gain_vs_external_pct"] = 100.0 * (summary["external_best_mae"] - summary["mae_mean"]) / summary["external_best_mae"]
    save_csv(raw, out, "error_decomposition_raw.csv")
    save_csv(summary, out, "error_decomposition.csv")
    fixed_pred = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    save_csv(fixed_pred, out, "fixed_test_predictions.csv")
    return summary, fixed_pred


def radius_from_candidate(col: str) -> float | None:
    if "_R" not in col:
        return None
    try:
        return float(col.split("_R", 1)[1].split("_", 1)[0])
    except Exception:
        return None


def nested_candidate_pool(conf_df: pd.DataFrame, target: str) -> list[str]:
    cols = [c for c in conf_df.columns if c.startswith("sum_pc_")]
    if target == "sterimol_B5":
        pool = [c for c in cols if "_R" not in c and any(k in c for k in ["Bmax", "B95", "B90"])]
    elif target == "sterimol_L":
        pool = [c for c in cols if "_R" not in c and any(k in c for k in ["Lpos", "Lneg", "Labs"])]
    elif target == "sterimol_burB5":
        pool = []
        for c in cols:
            r = radius_from_candidate(c)
            if r is not None and 3.8 <= r <= 5.0 and any(k in c for k in ["touch_Bmax", "touch_B95", "center_Bmax", "center_B95"]):
                pool.append(c)
    elif target == "sterimol_burL":
        pool = []
        for c in cols:
            r = radius_from_candidate(c)
            if r is not None and 2.5 <= r <= 3.2 and any(k in c for k in ["touch_Lpos", "touch_Labs", "center_Lpos", "center_Labs"]):
                pool.append(c)
    else:
        pool = []
    fixed = FIXED_TARGETS[target]["candidate"]
    if fixed in conf_df.columns and fixed not in pool:
        pool.insert(0, fixed)
    return sorted(set(pool))


def build_nested_cleanroom_all_targets(out: Path, conf_df: pd.DataFrame, mol_df: pd.DataFrame) -> pd.DataFrame:
    all_pool = sorted(set(col for target in FIXED_TARGETS for col in nested_candidate_pool(conf_df, target)))
    if not all_pool:
        df = pd.DataFrame()
        save_csv(df, out, "table3b_nested_cleanroom_all_targets.csv")
        return df
    agg = aggregate_level(conf_df, all_pool, "boltzmann")
    raw_rows: list[dict[str, Any]] = []
    methods = ["raw", "affine", "huber", "ridge_quadratic", "isotonic"]
    for target, cfg in FIXED_TARGETS.items():
        y = y_for(mol_df, target)
        pool = [col for col in nested_candidate_pool(conf_df, target) if col in agg.columns]
        for seed in SEEDS:
            masks = split_masks(mol_df, seed)
            fit_ids = masks["fit"][masks["fit"]].index.intersection(agg.index)
            cal_ids = masks["cal"][masks["cal"]].index.intersection(agg.index)
            test_ids = masks["test"][masks["test"]].index.intersection(agg.index)
            best: dict[str, Any] | None = None
            for candidate in pool:
                for method in methods:
                    x_fit = agg.loc[fit_ids, candidate].to_numpy(dtype=float)
                    y_fit = y.loc[fit_ids].to_numpy(dtype=float)
                    x_cal = agg.loc[cal_ids, candidate].to_numpy(dtype=float)
                    y_cal = y.loc[cal_ids].to_numpy(dtype=float)
                    pred_cal = x_cal if method == "raw" else apply_calibrator(method, x_fit, y_fit, x_cal)
                    ok = np.isfinite(pred_cal) & np.isfinite(y_cal)
                    if not ok.any():
                        continue
                    cal_mae = float(np.mean(np.abs(pred_cal[ok] - y_cal[ok])))
                    if best is None or cal_mae < float(best["cal_mae"]):
                        best = {"candidate": candidate, "calibration": method, "cal_mae": cal_mae}
            if best is None:
                continue
            x_fit = agg.loc[fit_ids, best["candidate"]].to_numpy(dtype=float)
            y_fit = y.loc[fit_ids].to_numpy(dtype=float)
            x_test = agg.loc[test_ids, best["candidate"]].to_numpy(dtype=float)
            y_test = y.loc[test_ids].to_numpy(dtype=float)
            pred_test = x_test if best["calibration"] == "raw" else apply_calibrator(best["calibration"], x_fit, y_fit, x_test)
            ok = np.isfinite(pred_test) & np.isfinite(y_test)
            met = _metrics(y_test[ok], pred_test[ok]) if ok.any() else {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
            raw_rows.append(
                {
                    "target": target,
                    "target_short": cfg["short"],
                    "level": LEVEL,
                    "seed": seed,
                    "candidate": best["candidate"],
                    "calibration": best["calibration"],
                    "cal_mae": best["cal_mae"],
                    "test_used_for_selection": False,
                    "selection_scope": "fit_train_only_select_on_cal_then_test_once",
                    **met,
                }
            )
    raw = pd.DataFrame(raw_rows)
    summary = summarize_evaluations(
        raw_rows,
        by=["target", "target_short", "level", "candidate", "calibration", "test_used_for_selection", "selection_scope"],
    )
    target_summary = summarize_evaluations(
        raw_rows,
        by=["target", "target_short", "level", "test_used_for_selection", "selection_scope"],
    )
    save_csv(raw, out, "nested_candidates_raw.csv")
    save_csv(summary, out, "nested_candidates_5seed.csv")
    save_csv(target_summary, out, "nested_target_summary.csv")
    return summary


def build_radius_sensitivity(out: Path, conf_df: pd.DataFrame, mol_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sweeps = {
        "sterimol_burB5": [round(x, 1) for x in np.arange(3.8, 5.01, 0.2)],
        "sterimol_burL": [round(x, 1) for x in np.arange(2.5, 3.21, 0.1)],
    }
    needed_cols = []
    for target, radii in sweeps.items():
        short = FIXED_TARGETS[target]["short"]
        suffix = "Bmax" if target.endswith("B5") else "Lpos"
        for radius in radii:
            col = f"sum_pc_R{radius:g}_touch_{suffix}"
            if col in conf_df.columns:
                needed_cols.append(col)
            else:
                rows.append(
                    {
                        "target": target,
                        "target_short": short,
                        "radius": radius,
                        "candidate": col,
                        "status": "missing_candidate_column",
                        "mae_mean": float("nan"),
                        "mae_std": float("nan"),
                    }
                )
    needed_cols = sorted(set(needed_cols))
    agg = aggregate_level(conf_df, needed_cols, "boltzmann") if needed_cols else pd.DataFrame()
    for target, radii in sweeps.items():
        cfg = FIXED_TARGETS[target]
        suffix = "Bmax" if target.endswith("B5") else "Lpos"
        for radius in radii:
            col = f"sum_pc_R{radius:g}_touch_{suffix}"
            if col not in agg.columns:
                continue
            for seed in SEEDS:
                met, _ = evaluate_table(agg, mol_df, target, col, cfg["calibration"], seed)
                rows.append(
                    {
                        "target": target,
                        "target_short": cfg["short"],
                        "radius": radius,
                        "candidate": col,
                        "status": "evaluated",
                        "seed": seed,
                        **met,
                    }
                )
    raw = pd.DataFrame(rows)
    summary = summarize_evaluations(
        raw[raw.get("status", "").eq("evaluated")].to_dict("records") if "status" in raw else [],
        by=["target", "target_short", "radius", "candidate", "status"],
    )
    missing = raw[raw.get("status", "").eq("missing_candidate_column")] if "status" in raw else pd.DataFrame()
    if not missing.empty:
        summary = pd.concat([summary, missing[["target", "target_short", "radius", "candidate", "status", "mae_mean", "mae_std"]]], ignore_index=True)
    save_csv(raw, out, "radius_sensitivity_raw.csv")
    save_csv(summary, out, "radius_sensitivity.csv")
    return summary


def build_calibration_learning_curve(out: Path, conf_df: pd.DataFrame, mol_df: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate_level(conf_df, fixed_candidate_cols(), "boltzmann")
    rows: list[dict[str, Any]] = []
    fit_sizes: list[int | None] = [0, 5, 10, 25, 50, 100, 250, 500, None]
    for target, cfg in FIXED_TARGETS.items():
        for fit_n in fit_sizes:
            for seed in SEEDS:
                raw = fit_n == 0
                sample_seed = 100000 + seed * 997 + (-1 if fit_n is None else int(fit_n))
                met, _ = evaluate_table(
                    agg,
                    mol_df,
                    target,
                    cfg["candidate"],
                    cfg["calibration"],
                    seed,
                    fit_n=None if fit_n is None or fit_n == 0 else int(fit_n),
                    sample_seed=sample_seed,
                    raw=raw,
                )
                rows.append(
                    {
                        "target": target,
                        "target_short": cfg["short"],
                        "candidate": cfg["candidate"],
                        "calibration": "raw" if raw else cfg["calibration"],
                        "calibration_fit_molecules": "raw_0" if fit_n == 0 else ("full" if fit_n is None else str(fit_n)),
                        "seed": seed,
                        **met,
                    }
                )
    raw = pd.DataFrame(rows)
    summary = summarize_evaluations(
        rows,
        by=["target", "target_short", "candidate", "calibration", "calibration_fit_molecules"],
    )
    save_csv(raw, out, "calibration_size_raw.csv")
    save_csv(summary, out, "calibration_size.csv")
    return summary


def clean_train_calibrators(
    clean_agg: pd.DataFrame, mol_df: pd.DataFrame
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str, str]]:
    calibrators: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str, str]] = {}
    for target, cfg in FIXED_TARGETS.items():
        y = y_for(mol_df, target)
        for seed in SEEDS:
            masks = split_masks(mol_df, seed)
            fit_ids = masks["fit"][masks["fit"]].index.intersection(clean_agg.index)
            x_fit = clean_agg.loc[fit_ids, cfg["candidate"]].to_numpy(dtype=float)
            y_fit = y.loc[fit_ids].to_numpy(dtype=float)
            calibrators[(target, seed)] = (x_fit, y_fit, cfg["candidate"], cfg["calibration"])
    return calibrators


def eval_perturbed_agg(
    agg: pd.DataFrame,
    mol_df: pd.DataFrame,
    calibrators: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str, str]],
    perturbation_type: str,
    parameter: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target, cfg in FIXED_TARGETS.items():
        y = y_for(mol_df, target)
        for seed in SEEDS:
            masks = split_masks(mol_df, seed)
            test_ids = masks["test"][masks["test"]].index.intersection(agg.index)
            x_fit, y_fit, candidate, method = calibrators[(target, seed)]
            x_test = agg.loc[test_ids, candidate].to_numpy(dtype=float)
            y_test = y.loc[test_ids].to_numpy(dtype=float)
            pred = apply_calibrator(method, x_fit, y_fit, x_test)
            ok = np.isfinite(pred) & np.isfinite(y_test)
            met = _metrics(y_test[ok], pred[ok]) if ok.any() else {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
            rows.append(
                {
                    "perturbation_type": perturbation_type,
                    "parameter": parameter,
                    "target": target,
                    "target_short": cfg["short"],
                    "seed": seed,
                    "candidate": candidate,
                    "calibration_trained_on": "clean_boltzmann_train",
                    **met,
                }
            )
    return rows


def load_raw_records(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("Kraken.pickle") as handle:
            return pickle.load(handle)


def build_conf_records(raw: dict[str, Any]) -> list[ConfRecord]:
    periodic = Chem.GetPeriodicTable()
    records: list[ConfRecord] = []
    for mol_id, record in sorted(raw.items(), key=lambda item: str(item[0])):
        smiles, mol_targets, conformer_dict = _record_parts(record)
        conf_pack: list[tuple[str, float, Any]] = []
        for conf_id, conf_record in conformer_dict.items():
            _sdf, weight, _conf_targets = conf_record
            conf_pack.append((str(conf_id), float(weight), conf_record))
        conf_pack.sort(key=lambda item: item[1], reverse=True)
        for rank, (conf_id, weight, conf_record) in enumerate(conf_pack):
            sdf_block, _weight, conf_targets = conf_record
            mol = _parse_sdf_mol(sdf_block)
            if mol is None:
                continue
            conf = mol.GetConformer()
            coords = np.asarray(conf.GetPositions(), dtype=np.float64)
            atomic_nums = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int16)
            p_indices = np.flatnonzero(atomic_nums == 15)
            p_idx = int(p_indices[0]) if p_indices.size else 0
            try:
                neighbor_indices = [int(n.GetIdx()) for n in mol.GetAtomWithIdx(p_idx).GetNeighbors()]
            except Exception:
                neighbor_indices = []
            vdw = np.asarray([float(periodic.GetRvdw(int(z))) for z in atomic_nums], dtype=np.float64)
            records.append(
                ConfRecord(
                    molecule_id=str(mol_id),
                    smiles=str(smiles),
                    conf_id=conf_id,
                    rank=rank,
                    weight=float(weight),
                    coords=coords,
                    atomic_nums=atomic_nums,
                    vdw=vdw,
                    heavy=atomic_nums > 1,
                    p_idx=p_idx,
                    neighbor_indices=neighbor_indices,
                    true_conf={target: float(conf_targets[target]) for target in KRAKEN_TARGETS},
                    true_mol={target: float(mol_targets[target]) for target in KRAKEN_TARGETS},
                )
            )
    return records


def sum_pc_axis(coords: np.ndarray, p_idx: int, neighbor_indices: list[int], weighted: str = "unit") -> np.ndarray:
    center = coords[p_idx]
    vecs = []
    for idx in neighbor_indices[:4]:
        raw = coords[int(idx)] - center
        norm = float(np.linalg.norm(raw))
        if norm < 1e-12:
            continue
        if weighted == "unit":
            vecs.append(raw / norm)
        elif weighted == "bond_length":
            vecs.append(raw)
        elif weighted == "inverse_bond_length":
            vecs.append(raw / (norm * norm))
    if not vecs:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    return _safe_unit(np.sum(vecs, axis=0))


def estimator_axes(rec: ConfRecord, coords: np.ndarray) -> dict[str, np.ndarray]:
    axes: dict[str, np.ndarray] = {}
    axes["sum_pc_unit"] = sum_pc_axis(coords, rec.p_idx, rec.neighbor_indices, "unit")
    axes["raw_vector_sum"] = sum_pc_axis(coords, rec.p_idx, rec.neighbor_indices, "bond_length")
    axes["inverse_bond_weighted"] = sum_pc_axis(coords, rec.p_idx, rec.neighbor_indices, "inverse_bond_length")
    center = coords[rec.p_idx]
    for i, idx in enumerate(rec.neighbor_indices[:4]):
        axis = _safe_unit(coords[int(idx)] - center)
        if np.linalg.norm(axis) > 0:
            axes[f"pc{i}"] = axis
    heavy_rel = coords[rec.heavy] - center
    if len(heavy_rel) >= 3:
        cov = (heavy_rel.T @ heavy_rel) / max(float(len(heavy_rel)), 1.0)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        for rank, idx in enumerate(order):
            axes[f"pca{rank}"] = _safe_unit(vecs[:, int(idx)])
        axes["best_fit_plane_normal"] = _safe_unit(vecs[:, int(order[-1])])
    return {name: axis for name, axis in axes.items() if np.linalg.norm(axis) > 0}


def fixed_formula_row_from_record(rec: ConfRecord, sigma: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Any]:
    coords = rec.coords.copy()
    if sigma > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        coords = coords + rng.normal(0.0, sigma, size=coords.shape)
    center = coords[rec.p_idx]
    axis = sum_pc_axis(coords, rec.p_idx, rec.neighbor_indices, "unit")
    feats = _sterimol_features_for_axis(coords, center, rec.vdw, rec.heavy, axis, np.asarray([2.7, 4.4], dtype=float))
    return {
        "molecule_id": rec.molecule_id,
        "conf_id": rec.conf_id,
        "rank": rec.rank,
        "weight": rec.weight,
        "sum_pc_Bmax_all": feats.get("Bmax_all", float("nan")),
        "sum_pc_Lpos_all": feats.get("Lpos_all", float("nan")),
        "sum_pc_R4.4_touch_Bmax": feats.get("R4.4_touch_Bmax", float("nan")),
        "sum_pc_R2.7_touch_Lpos": feats.get("R2.7_touch_Lpos", float("nan")),
    }


def build_geometry_rows(records: list[ConfRecord], sigma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [fixed_formula_row_from_record(rec, sigma=sigma, rng=rng) for rec in records]
    return pd.DataFrame(rows)


def build_perturbation_robustness(
    out: Path,
    conf_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    records: list[ConfRecord] | None,
    skip_geometry_noise: bool,
) -> pd.DataFrame:
    fixed_cols = fixed_candidate_cols()
    clean_agg = aggregate_level(conf_df, fixed_cols, "boltzmann")
    calibrators = clean_train_calibrators(clean_agg, mol_df)
    rows: list[dict[str, Any]] = []
    rows.extend(eval_perturbed_agg(clean_agg, mol_df, calibrators, "clean", "boltzmann_298K"))

    for temp in [250.0, 273.15, 298.15, 323.15, 350.0, 400.0]:
        agg = aggregate_custom_weights(conf_df, fixed_cols, "temperature", parameter=temp)
        rows.extend(eval_perturbed_agg(agg, mol_df, calibrators, "temperature_K", f"{temp:g}"))

    for sigma in [0.05, 0.10, 0.25, 0.50, 1.00]:
        for rep in [0, 1, 2]:
            agg = aggregate_custom_weights(conf_df, fixed_cols, "energy_noise", parameter=sigma, rng_seed=7713 + rep)
            rows.extend(eval_perturbed_agg(agg, mol_df, calibrators, "energy_noise_kcal_mol_sd", f"{sigma:g}_rep{rep}"))

    if not skip_geometry_noise and records is not None:
        for sigma in [0.01, 0.03, 0.05, 0.10]:
            for rep in [0, 1]:
                geom_df = build_geometry_rows(records, sigma=sigma, seed=8800 + rep)
                agg = aggregate_level(geom_df, fixed_cols, "boltzmann")
                rows.extend(eval_perturbed_agg(agg, mol_df, calibrators, "geometry_noise_A_sd", f"{sigma:g}_rep{rep}"))

    raw = pd.DataFrame(rows)
    summary = summarize_evaluations(
        rows,
        by=["perturbation_type", "parameter", "target", "target_short", "candidate", "calibration_trained_on"],
    )
    clean = summary[summary["perturbation_type"].eq("clean")][["target", "mae_mean"]].rename(columns={"mae_mean": "clean_mae_mean"})
    summary = summary.merge(clean, on="target", how="left")
    summary["delta_vs_clean_mae"] = summary["mae_mean"] - summary["clean_mae_mean"]
    save_csv(raw, out, "perturbation_raw.csv")
    save_csv(summary, out, "perturbation_5seed_full.csv")
    return summary


def fibonacci_sphere(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    points = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / float(max(n - 1, 1))) * 2.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        points.append([math.cos(theta) * radius, y, math.sin(theta) * radius])
    return np.asarray(points, dtype=float)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    au = _safe_unit(a)
    bu = _safe_unit(b)
    dot = float(abs(np.dot(au, bu)))
    dot = min(1.0, max(0.0, dot))
    return float(math.degrees(math.acos(dot)))


def descriptor_error_for_axis(rec: ConfRecord, axis: np.ndarray, scale_b: float, scale_l: float) -> tuple[float, dict[str, float]]:
    center = rec.coords[rec.p_idx]
    feats = _sterimol_features_for_axis(rec.coords, center, rec.vdw, rec.heavy, axis, np.asarray([], dtype=float))
    b_err = abs(float(feats["Bmax_all"]) - rec.true_conf["sterimol_B5"])
    l_err = min(abs(float(feats["Lpos_all"]) - rec.true_conf["sterimol_L"]), abs(float(feats["Lneg_all"]) - rec.true_conf["sterimol_L"]))
    score = b_err / max(scale_b, 1e-8) + l_err / max(scale_l, 1e-8)
    return float(score), {"B5_abs_error": float(b_err), "L_abs_error": float(l_err)}


def build_axis_diagnostic(out: Path, records: list[ConfRecord], sphere_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    true_b = np.asarray([rec.true_conf["sterimol_B5"] for rec in records], dtype=float)
    true_l = np.asarray([rec.true_conf["sterimol_L"] for rec in records], dtype=float)
    scale_b = float(np.nanstd(true_b)) or 1.0
    scale_l = float(np.nanstd(true_l)) or 1.0
    sphere = fibonacci_sphere(sphere_n)

    angle_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for rec in records:
        axes = estimator_axes(rec, rec.coords)
        axis_names = list(axes.keys())
        axis_arr = np.vstack([axes[name] for name in axis_names]).astype(float)
        dirs = np.vstack([axis_arr, sphere]).astype(float)
        norms = np.linalg.norm(dirs, axis=1)
        dirs = dirs[norms > 1e-12]
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

        center = rec.coords[rec.p_idx]
        rel = rec.coords - center[None, :]
        proj = rel @ dirs.T
        dist2 = np.sum(rel * rel, axis=1)[:, None]
        perp = np.sqrt(np.maximum(0.0, dist2 - proj * proj))
        radii = rec.vdw[:, None]
        bmax = np.max(perp + radii, axis=0)
        lpos = np.max(proj + radii, axis=0)
        lneg = np.max(-proj + radii, axis=0)
        b_err = np.abs(bmax - rec.true_conf["sterimol_B5"])
        l_err = np.minimum(
            np.abs(lpos - rec.true_conf["sterimol_L"]),
            np.abs(lneg - rec.true_conf["sterimol_L"]),
        )
        scores = b_err / max(scale_b, 1e-8) + l_err / max(scale_l, 1e-8)
        best_idx = int(np.nanargmin(scores))
        best_axis = dirs[best_idx]
        best_score = float(scores[best_idx])

        for i, name in enumerate(axis_names):
            axis = axis_arr[i]
            score = float(scores[i])
            errs = {"B5_abs_error": float(b_err[i]), "L_abs_error": float(l_err[i])}
            angle_rows.append(
                {
                    "molecule_id": rec.molecule_id,
                    "conf_id": rec.conf_id,
                    "estimator": name,
                    "angle_deg_to_descriptor_inferred_reference": angle_deg(axis, best_axis),
                    "descriptor_inferred_score": best_score,
                    "estimator_score": score,
                    **errs,
                }
            )
            error_rows.append(
                {
                    "molecule_id": rec.molecule_id,
                    "conf_id": rec.conf_id,
                    "estimator": name,
                    "B5_abs_error": errs["B5_abs_error"],
                    "L_abs_error": errs["L_abs_error"],
                    "angle_deg": angle_deg(axis, best_axis),
                }
            )
    angle_df = pd.DataFrame(angle_rows)
    summary_rows = []
    for estimator, group in angle_df.groupby("estimator", sort=True):
        vals = group["angle_deg_to_descriptor_inferred_reference"].to_numpy(dtype=float)
        desc = numeric_summary(vals)
        summary_rows.append(
            {
                "reference_type": "descriptor_inferred_from_true_conformer_B5_L",
                "not_used_for_prediction": True,
                "estimator": estimator,
                **{f"angle_deg_{k}": v for k, v in desc.items()},
                "pct_angle_gt_5": float((vals > 5).mean() * 100.0),
                "pct_angle_gt_10": float((vals > 10).mean() * 100.0),
                "pct_angle_gt_20": float((vals > 20).mean() * 100.0),
                "n_conformers": int(len(group)),
            }
        )
    summary = pd.DataFrame(summary_rows)

    corr_rows = []
    for estimator, group in pd.DataFrame(error_rows).groupby("estimator", sort=True):
        angle_rank = group["angle_deg"].rank(method="average")
        for target_col, target_name in [("B5_abs_error", "sterimol_B5"), ("L_abs_error", "sterimol_L")]:
            err = group[target_col].astype(float)
            err_rank = err.rank(method="average")
            corr_rows.append(
                {
                    "reference_type": "descriptor_inferred_from_true_conformer_B5_L",
                    "not_used_for_prediction": True,
                    "estimator": estimator,
                    "target": target_name,
                    "pearson_angle_vs_abs_error": float(group["angle_deg"].corr(err)),
                    "spearman_angle_vs_abs_error": float(angle_rank.corr(err_rank)),
                    "n_conformers": int(len(group)),
                }
            )
    corr = pd.DataFrame(corr_rows)
    save_csv(summary, out, "table1_descriptor_inferred_axis_angular_summary.csv")
    save_csv(corr, out, "table1b_axis_error_correlation.csv")
    save_csv(angle_df, out, "table1c_axis_diagnostic_raw_by_conformer.csv")
    return summary, corr


def classify_family(smiles: str, n_confs: int) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ligand_family": "invalid_smiles", "p_neighbor_pattern": "unknown", "ring_at_p_neighbor": False, "rotatable_bonds": float("nan"), "conformer_bucket": "unknown"}
    p_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 15]
    if not p_atoms:
        pattern = "no_P"
        ring_neighbor = False
    else:
        p_atom = p_atoms[0]
        neighbors = list(p_atom.GetNeighbors())
        aromatic = sum(1 for atom in neighbors if atom.GetIsAromatic())
        carbon = sum(1 for atom in neighbors if atom.GetAtomicNum() == 6)
        ring_neighbor = any(atom.IsInRing() for atom in neighbors)
        if carbon < 3:
            pattern = "non_tri_carbon_P"
        elif aromatic == 3:
            pattern = "triaryl"
        elif aromatic == 0:
            pattern = "trialkyl"
        else:
            pattern = "mixed_aryl_alkyl"
    rot = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    if n_confs <= 4:
        bucket = "1_4_confs"
    elif n_confs <= 8:
        bucket = "5_8_confs"
    elif n_confs <= 16:
        bucket = "9_16_confs"
    else:
        bucket = "17plus_confs"
    family = pattern
    if ring_neighbor:
        family = f"{family}_ring"
    return {
        "ligand_family": family,
        "p_neighbor_pattern": pattern,
        "ring_at_p_neighbor": bool(ring_neighbor),
        "rotatable_bonds": rot,
        "conformer_bucket": bucket,
    }


def build_chemical_family_stratification(out: Path, mol_df: pd.DataFrame, conf_df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        empty = pd.DataFrame()
        save_csv(empty, out, "table7_chemical_family_stratification.csv")
        return empty
    n_confs = conf_df.groupby("molecule_id").size().astype(int).to_dict()
    family_rows = []
    for row in mol_df.itertuples(index=False):
        mid = str(getattr(row, "molecule_id"))
        family = classify_family(str(getattr(row, "smiles")), int(n_confs.get(mid, 0)))
        family["molecule_id"] = mid
        family_rows.append(family)
    family_df = pd.DataFrame(family_rows)
    merged = predictions.merge(family_df, on="molecule_id", how="left")
    seed_family_rows = []
    group_cols = ["target", "seed", "ligand_family"]
    for keys, group in merged.groupby(group_cols, dropna=False, sort=True):
        target, seed, family = keys
        seed_family_rows.append(
            {
                "target": target,
                "target_short": FIXED_TARGETS[target]["short"],
                "seed": int(seed),
                "ligand_family": family,
                "n_test_molecules": int(group["molecule_id"].nunique()),
                "mae": float(group["abs_error"].mean()),
                "median_abs_error": float(group["abs_error"].median()),
            }
        )
    raw = pd.DataFrame(seed_family_rows)
    summary = summarize_evaluations(raw.to_dict("records"), by=["target", "target_short", "ligand_family"])
    if not raw.empty:
        counts = raw.groupby(["target", "ligand_family"], as_index=False)["n_test_molecules"].mean().rename(columns={"n_test_molecules": "mean_test_molecules_per_seed"})
        summary = summary.merge(counts, on=["target", "ligand_family"], how="left")
    save_csv(raw, out, "chemical_families_raw.csv")
    save_csv(summary, out, "chemical_families.csv")
    return summary


def make_markdown_summary(
    out: Path,
    error_decomp: pd.DataFrame,
    radius: pd.DataFrame,
    learning: pd.DataFrame,
    perturb: pd.DataFrame,
    family: pd.DataFrame,
    axis: pd.DataFrame | None,
) -> None:
    lines = [
        "# FCSR Kraken Analysis",
        "",
        "This report was generated from the released Kraken archive and a locally generated feature cache.",
        "Exact hidden DFT metal-P reference-axis vectors are not present in the released pickle; axis rows marked descriptor-inferred are diagnostics only.",
        "",
    ]
    lines.extend(["", "## Main Error Decomposition", ""])
    keep = error_decomp[error_decomp["stage"].isin(["oracle_conformer_labels_boltzmann_all_confs", "raw_formula_boltzmann", "calibrated_formula_boltzmann", "calibrated_formula_uniform"])]
    cols = ["target_short", "stage", "mae_mean", "mae_std", "relative_gain_vs_external_pct"]
    lines.extend(keep[cols].to_markdown(index=False, floatfmt=".6f").splitlines())
    lines.extend(["", "## Radius Sensitivity", ""])
    if not radius.empty:
        best_radius = radius.dropna(subset=["mae_mean"]).sort_values(["target", "mae_mean"]).groupby("target", as_index=False).head(3)
        lines.extend(best_radius[["target_short", "radius", "mae_mean", "mae_std"]].to_markdown(index=False, floatfmt=".6f").splitlines())
    lines.extend(["", "## Calibration Learning Curve", ""])
    if not learning.empty:
        view = learning[learning["calibration_fit_molecules"].isin(["raw_0", "10", "50", "100", "500", "full"])]
        lines.extend(view[["target_short", "calibration_fit_molecules", "mae_mean", "mae_std"]].to_markdown(index=False, floatfmt=".6f").splitlines())
    lines.extend(["", "## Robustness Snapshot", ""])
    if not perturb.empty:
        snap = perturb[perturb["perturbation_type"].isin(["clean", "temperature_K", "energy_noise_kcal_mol_sd", "geometry_noise_A_sd"])].copy()
        snap = snap.sort_values(["target", "perturbation_type", "mae_mean"]).groupby(["target", "perturbation_type"], as_index=False).head(2)
        lines.extend(snap[["target_short", "perturbation_type", "parameter", "mae_mean", "delta_vs_clean_mae"]].to_markdown(index=False, floatfmt=".6f").splitlines())
    if axis is not None and not axis.empty:
        lines.extend(["", "## Axis Diagnostic", ""])
        view = axis[axis["estimator"].isin(["sum_pc_unit", "raw_vector_sum", "inverse_bond_weighted", "pca0", "best_fit_plane_normal"])]
        lines.extend(view[["estimator", "angle_deg_median", "angle_deg_p90", "pct_angle_gt_10"]].to_markdown(index=False, floatfmt=".4f").splitlines())
    lines.extend(["", "## Chemical Families", ""])
    if not family.empty:
        view = family.sort_values(["target", "mae_mean"]).groupby("target", as_index=False).head(5)
        lines.extend(view[["target_short", "ligand_family", "mae_mean", "mae_std", "mean_test_molecules_per_seed"]].to_markdown(index=False, floatfmt=".6f").splitlines())
    path = out / "ANALYSIS_SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {path}")


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.out)
    print(f"[info] output={out}")
    mol_df, _meta = load_kraken(args.kraken_zip, download=False)
    mol_df["molecule_id"] = mol_df["molecule_id"].astype(str)
    conf_df = load_formula_cache(args.formula_cache)
    print(f"[info] molecules={len(mol_df)} conformer_rows={len(conf_df)}")

    missing_cols = [col for col in fixed_candidate_cols() if col not in conf_df.columns]
    if missing_cols:
        raise SystemExit(f"Missing fixed candidate columns in {args.formula_cache}: {missing_cols}")

    error_decomp, predictions = build_error_decomposition(out, conf_df, mol_df)
    nested_cleanroom = build_nested_cleanroom_all_targets(out, conf_df, mol_df)
    _ = nested_cleanroom
    radius = build_radius_sensitivity(out, conf_df, mol_df)
    learning = build_calibration_learning_curve(out, conf_df, mol_df)

    records: list[ConfRecord] | None = None
    axis_summary: pd.DataFrame | None = None
    if not args.skip_axis_diagnostic or not args.skip_geometry_noise:
        print("[info] loading raw Kraken conformers once for geometry/axis diagnostics")
        raw = load_raw_records(args.kraken_zip)
        records = build_conf_records(raw)
        print(f"[info] parsed_conformer_records={len(records)}")

    if not args.skip_axis_diagnostic and records is not None:
        axis_summary, _corr = build_axis_diagnostic(out, records, args.axis_sphere)

    perturb = build_perturbation_robustness(out, conf_df, mol_df, records, args.skip_geometry_noise)
    compact = perturb.rename(columns={"parameter": "magnitude"})[
        ["perturbation_type", "magnitude", "target_short", "mae_mean", "mae_std", "n_seeds"]
    ]
    save_csv(compact, out, "perturbation_5seed.csv")
    family = build_chemical_family_stratification(out, mol_df, conf_df, predictions)
    make_markdown_summary(out, error_decomp, radius, learning, perturb, family, axis_summary)
    print("[done] supplement audit complete")


if __name__ == "__main__":
    main()

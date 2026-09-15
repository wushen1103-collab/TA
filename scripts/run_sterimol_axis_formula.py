#!/usr/bin/env python
"""Axis-recovered Sterimol formula baseline for Kraken.

The Kraken Sterimol targets are geometric quantities. This script reconstructs
the phosphine ligand axis from its bonded substituent directions, evaluates
DBSTEP-like analytic Sterimol candidates on each conformer, aggregates top-k
conformers with their Boltzmann weights, and uses only train/validation/cal
molecule labels to choose/calibrate the formula.

No test conformer component labels are used for prediction.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fcsr.kraken import KRAKEN_TARGETS, _parse_sdf_mol, _record_parts, load_kraken  # noqa: E402
from fcsr.splits import scaffold_split  # noqa: E402
from fcsr.utils import ensure_dir, write_json  # noqa: E402

LEVEL_K = {"L2": 1, "L3_4": 4, "L3_8": 8, "L4_32": 32}
DEFAULT_RADII = np.round(np.arange(2.5, 5.01, 0.1), 1).astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kraken-zip", type=Path, default=Path("data/Kraken.zip"))
    parser.add_argument("--targets", nargs="+", default=KRAKEN_TARGETS)
    parser.add_argument("--levels", nargs="+", default=list(LEVEL_K))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--split-mode", choices=["scaffold", "random"], default="random")
    parser.add_argument(
        "--split-fractions",
        nargs=4,
        type=float,
        default=[0.7, 0.0, 0.1, 0.2],
        metavar=("TRAIN", "VAL", "CAL", "TEST"),
        help="Fractions assigned to train/val/cal/test. The fitting mask uses train+val; use 0.7 0 0.1 0.2 for MARCEL-style 70/10/20 with validation as calibration.",
    )
    parser.add_argument("--out", type=Path, default=Path("results/candidate_search"))
    parser.add_argument("--cache", type=Path, default=Path("data/formula_rows.joblib"))
    parser.add_argument("--radii", nargs="+", type=float, default=DEFAULT_RADII.tolist())
    parser.add_argument("--axis-mode", choices=["sum_pc", "sum_and_pca", "all"], default="sum_pc")
    parser.add_argument("--weight-mode", choices=["boltzmann", "uniform"], default="boltzmann")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--features-only", action="store_true", help="Build analytic conformer cache without selecting on test labels.")
    parser.add_argument("--include-isotonic", action="store_true")
    return parser.parse_args()


def _safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return vec / norm


def _random_split(df: pd.DataFrame, seed: int, fractions: tuple[float, float, float, float]) -> pd.Series:
    rng = np.random.default_rng(seed)
    order = np.arange(len(df))
    rng.shuffle(order)
    n = len(df)
    cuts = np.cumsum(np.asarray(fractions) * n).astype(int)
    split = np.empty(n, dtype=object)
    split[order[: cuts[0]]] = "train"
    split[order[cuts[0] : cuts[1]]] = "val"
    split[order[cuts[1] : cuts[2]]] = "cal"
    split[order[cuts[2] :]] = "test"
    return pd.Series(split, index=df.index, name="split")


def _axis_candidates(mol: Chem.Mol, axis_mode: str) -> dict[str, tuple[np.ndarray, str]]:
    conf = mol.GetConformer()
    coords = np.asarray(conf.GetPositions(), dtype=np.float64)
    atomic_nums = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int16)
    p_indices = np.flatnonzero(atomic_nums == 15)
    p_idx = int(p_indices[0]) if p_indices.size else 0
    center = coords[p_idx]

    axes: dict[str, tuple[np.ndarray, str]] = {}
    neighbor_units = []
    try:
        neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(p_idx).GetNeighbors()]
    except Exception:
        neighbors = []
    for n_idx, atom_idx in enumerate(neighbors[:4]):
        unit = _safe_unit(coords[int(atom_idx)] - center)
        if np.linalg.norm(unit) > 0:
            neighbor_units.append(unit)
            if axis_mode == "all":
                axes[f"pc{n_idx}"] = (unit, "P-C bond axis")
    if neighbor_units:
        summed = _safe_unit(np.sum(neighbor_units, axis=0))
        if np.linalg.norm(summed) > 0:
            axes["sum_pc"] = (summed, "sum of P-C unit vectors")
            if axis_mode in {"sum_and_pca", "all"}:
                axes["lone_pair"] = (-summed, "opposite of summed P-C unit vectors")

    if axis_mode in {"sum_and_pca", "all"}:
        heavy = atomic_nums > 1
        rel_heavy = coords[heavy] - center
        if len(rel_heavy) >= 3:
            cov = (rel_heavy.T @ rel_heavy) / max(float(len(rel_heavy)), 1.0)
            vals, vecs = np.linalg.eigh(cov)
            for rank, idx in enumerate(np.argsort(vals)[::-1]):
                axis = _safe_unit(vecs[:, int(idx)])
                if np.linalg.norm(axis) > 0:
                    axes[f"pca{rank}"] = (axis, "principal axis of heavy atom coordinates")
    if axis_mode == "all":
        axes["global_x"] = (np.asarray([1.0, 0.0, 0.0], dtype=np.float64), "global x-axis control")
    if not axes:
        axes["fallback_x"] = (np.asarray([1.0, 0.0, 0.0], dtype=np.float64), "fallback x-axis")
    return axes


def _sterimol_features_for_axis(
    coords: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    heavy: np.ndarray,
    axis: np.ndarray,
    radii_scan: np.ndarray,
) -> dict[str, float]:
    unit = _safe_unit(axis)
    rel = coords - center[None, :]
    proj = rel @ unit
    dist2 = np.sum(rel * rel, axis=1)
    perp = np.sqrt(np.maximum(0.0, dist2 - proj * proj))
    dist = np.sqrt(np.maximum(0.0, dist2))

    b_surface = perp + radii
    l_pos = proj + radii
    l_neg = -proj + radii
    l_abs = np.abs(proj) + radii
    out = {
        "Bmax_all": float(np.max(b_surface)),
        "B95_all": float(np.quantile(b_surface, 0.95)),
        "B90_all": float(np.quantile(b_surface, 0.90)),
        "Bmax_heavy": float(np.max(b_surface[heavy])) if heavy.any() else float(np.max(b_surface)),
        "B95_heavy": float(np.quantile(b_surface[heavy], 0.95)) if heavy.any() else float(np.quantile(b_surface, 0.95)),
        "Lpos_all": float(np.max(l_pos)),
        "Lneg_all": float(np.max(l_neg)),
        "Labs_all": float(np.max(l_abs)),
        "Lpos_heavy": float(np.max(l_pos[heavy])) if heavy.any() else float(np.max(l_pos)),
        "Labs_heavy": float(np.max(l_abs[heavy])) if heavy.any() else float(np.max(l_abs)),
    }

    for radius in radii_scan:
        radius_tag = f"{radius:g}"
        center_mask = dist <= radius
        touch_mask = (dist - radii) <= radius
        inner_mask = (dist + radii) <= radius
        for mask_name, mask in [("center", center_mask), ("touch", touch_mask), ("inner", inner_mask)]:
            if not mask.any():
                continue
            b_masked = b_surface[mask]
            lpos_masked = l_pos[mask]
            labs_masked = l_abs[mask]
            out[f"R{radius_tag}_{mask_name}_Bmax"] = float(np.max(b_masked))
            out[f"R{radius_tag}_{mask_name}_B95"] = float(np.quantile(b_masked, 0.95))
            out[f"R{radius_tag}_{mask_name}_Lpos"] = float(np.max(lpos_masked))
            out[f"R{radius_tag}_{mask_name}_Labs"] = float(np.max(labs_masked))
    return out


def _feature_one(mol_id: str, record: Any, radii_scan: np.ndarray, axis_mode: str) -> list[dict[str, object]]:
    smiles, mol_targets, conformer_dict = _record_parts(record)
    periodic = Chem.GetPeriodicTable()
    rows: list[dict[str, object]] = []
    conf_rows = []
    for conf_id, conf_record in conformer_dict.items():
        sdf_block, weight, conf_targets = conf_record
        mol = _parse_sdf_mol(sdf_block)
        if mol is None:
            continue
        conf = mol.GetConformer()
        coords = np.asarray(conf.GetPositions(), dtype=np.float64)
        atomic_nums = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int16)
        p_indices = np.flatnonzero(atomic_nums == 15)
        p_idx = int(p_indices[0]) if p_indices.size else 0
        center = coords[p_idx]
        vdw = np.asarray([float(periodic.GetRvdw(int(z))) for z in atomic_nums], dtype=np.float64)
        heavy = atomic_nums > 1

        row: dict[str, object] = {
            "molecule_id": str(mol_id),
            "smiles": smiles,
            "conf_id": str(conf_id),
            "weight": float(weight),
        }
        for target in KRAKEN_TARGETS:
            row[f"{target}_true_mol"] = float(mol_targets[target])
            row[f"{target}_true_conf"] = float(conf_targets[target])

        for axis_name, (axis, _axis_note) in _axis_candidates(mol, axis_mode=axis_mode).items():
            feats = _sterimol_features_for_axis(coords, center, vdw, heavy, axis, radii_scan)
            for name, value in feats.items():
                row[f"{axis_name}_{name}"] = value
        conf_rows.append(row)

    conf_rows.sort(key=lambda item: float(item["weight"]), reverse=True)
    for rank, row in enumerate(conf_rows):
        row["rank"] = rank
        rows.append(row)
    return rows


def _load_formula_rows(args: argparse.Namespace) -> pd.DataFrame:
    if args.cache.exists() and not args.force_features:
        return joblib.load(args.cache)
    with zipfile.ZipFile(args.kraken_zip) as archive:
        with archive.open("Kraken.pickle") as handle:
            raw = pickle.load(handle)
    rows = []
    radii_scan = np.asarray(args.radii, dtype=np.float64)
    for mol_id, record in sorted(raw.items(), key=lambda item: str(item[0])):
        rows.extend(_feature_one(str(mol_id), record, radii_scan, args.axis_mode))
    df = pd.DataFrame(rows)
    ensure_dir(args.cache.parent)
    joblib.dump(df, args.cache)
    return df


def _target_candidate_cols(target: str, candidate_cols: list[str]) -> list[str]:
    if target.endswith("B5"):
        keywords = ("Bmax", "B95", "B90")
    else:
        keywords = ("Lpos", "Labs", "Lneg")
    preferred = []
    fallback = []
    for col in candidate_cols:
        if any(key in col for key in keywords):
            if target in {"sterimol_B5", "sterimol_L"} and col.startswith("sum_pc_"):
                preferred.append(col)
            else:
                fallback.append(col)
    return preferred + fallback


def _aggregate_formula_predictions(
    conf_df: pd.DataFrame,
    candidate_cols: list[str],
    levels: list[str],
    weight_mode: str = "boltzmann",
) -> dict[str, pd.DataFrame]:
    out = {level: [] for level in levels}
    for mol_id, group in conf_df.groupby("molecule_id", sort=True):
        group = group.sort_values("rank")
        values = group[candidate_cols].to_numpy(dtype=np.float64)
        if weight_mode == "uniform":
            weights = np.ones(len(group), dtype=np.float64)
        else:
            weights = group["weight"].to_numpy(dtype=np.float64)
        for level in levels:
            k = LEVEL_K[level]
            n = min(k, len(group))
            w = weights[:n]
            vals = values[:n]
            mask = np.isfinite(w) & (w > 0)
            if not mask.any():
                pred = np.nanmean(vals, axis=0)
            else:
                pred = np.sum(vals[mask] * w[mask, None], axis=0) / max(float(np.sum(w[mask])), 1e-12)
            out[level].append((mol_id, pred))
    return {
        level: pd.DataFrame([row[1] for row in rows], index=[row[0] for row in rows], columns=candidate_cols)
        for level, rows in out.items()
    }


def _calibrated_predictions(method: str, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    if method == "raw":
        return x_eval.astype(np.float64)
    if method == "affine":
        model = LinearRegression()
        model.fit(x_train[:, None], y_train)
        return model.predict(x_eval[:, None])
    if method == "ridge_quadratic":
        train_mat = np.column_stack([x_train, x_train * x_train])
        eval_mat = np.column_stack([x_eval, x_eval * x_eval])
        model = RidgeCV(alphas=[1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0])
        model.fit(train_mat, y_train)
        return model.predict(eval_mat)
    if method == "huber":
        model = HuberRegressor(alpha=1e-4, epsilon=1.35, max_iter=500)
        model.fit(x_train[:, None], y_train)
        return model.predict(x_eval[:, None])
    if method == "isotonic":
        increasing = np.corrcoef(x_train, y_train)[0, 1] >= 0
        model = IsotonicRegression(increasing=bool(increasing), out_of_bounds="clip")
        model.fit(x_train, y_train)
        return model.predict(x_eval)
    raise ValueError(f"Unknown calibration method: {method}")


def _metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    levels = [level for level in args.levels if level in LEVEL_K]
    if not levels:
        raise ValueError(f"No valid levels in {args.levels}; choose from {list(LEVEL_K)}")
    split_fractions = tuple(float(x) for x in args.split_fractions)
    if len(split_fractions) != 4 or abs(sum(split_fractions) - 1.0) > 1e-6:
        raise ValueError("--split-fractions must contain four values summing to 1.0")

    mol_df, meta = load_kraken(args.kraken_zip, download=False)
    formula_df = _load_formula_rows(args)
    if args.features_only:
        print(f"Built {len(formula_df)} conformer rows in {args.cache}")
        return
    excluded = {"molecule_id", "smiles", "conf_id", "weight", "rank"}
    excluded.update({f"{target}_true_mol" for target in KRAKEN_TARGETS})
    excluded.update({f"{target}_true_conf" for target in KRAKEN_TARGETS})
    candidate_cols = [col for col in formula_df.columns if col not in excluded]
    if not candidate_cols:
        raise RuntimeError("No formula candidate columns were built.")
    aggregated = _aggregate_formula_predictions(formula_df, candidate_cols, levels, weight_mode=args.weight_mode)

    calibration_methods = ["raw", "affine", "ridge_quadratic", "huber"]
    if args.include_isotonic:
        calibration_methods.append("isotonic")

    all_metrics = []
    selections = []
    predictions = []
    for seed in args.seeds:
        if args.split_mode == "scaffold":
            splits = scaffold_split(mol_df, seed=seed, fractions=split_fractions)
        else:
            splits = _random_split(mol_df, seed=seed, fractions=split_fractions)
        mol_seed = mol_df.copy()
        mol_seed["split"] = splits
        y_by_target = mol_seed.set_index("molecule_id")
        split_by_mol = mol_seed.set_index("molecule_id")["split"]
        train_mask_by_mol = split_by_mol.isin(["train", "val"])
        select_mask_by_mol = split_by_mol.eq("cal")
        test_mask_by_mol = split_by_mol.eq("test")

        for target in args.targets:
            target_cols = _target_candidate_cols(target, candidate_cols)
            y_all = y_by_target[target].astype(float)
            scored_for_target = []
            for level in levels:
                pred_table = aggregated[level].reindex(y_all.index)
                train_mask = train_mask_by_mol.reindex(y_all.index).to_numpy()
                select_mask = select_mask_by_mol.reindex(y_all.index).to_numpy()
                test_mask = test_mask_by_mol.reindex(y_all.index).to_numpy()
                y = y_all.to_numpy(dtype=np.float64)
                for col in target_cols:
                    x = pred_table[col].to_numpy(dtype=np.float64)
                    valid = np.isfinite(x) & np.isfinite(y)
                    for cal_method in calibration_methods:
                        fit_mask = train_mask & valid
                        cal_mask = select_mask & valid
                        eval_mask = test_mask & valid
                        if fit_mask.sum() < 5 or cal_mask.sum() < 5 or eval_mask.sum() < 5:
                            continue
                        calibrated = _calibrated_predictions(cal_method, x[fit_mask], y[fit_mask], x[valid])
                        pred_full = np.full_like(y, np.nan, dtype=np.float64)
                        pred_full[np.flatnonzero(valid)] = calibrated
                        cal_metrics = _metrics(y[cal_mask], pred_full[cal_mask])
                        row = {
                            "dataset": f"kraken_{target}",
                            "target": target,
                            "seed": seed,
                            "split_mode": args.split_mode,
                            "method": "axis_formula",
                            "level": level,
                            "avg_cost_seconds": float(LEVEL_K[level]),
                            "candidate": col,
                            "calibration": cal_method,
                            "cal_mae": cal_metrics["mae"],
                            "cal_rmse": cal_metrics["rmse"],
                            "cal_r2": cal_metrics["r2"],
                            "mae": float("nan"),
                            "rmse": float("nan"),
                            "r2": float("nan"),
                            "n_train": int(fit_mask.sum()),
                            "n_cal": int(cal_mask.sum()),
                            "n_test": int(eval_mask.sum()),
                        }
                        all_metrics.append(row)
                        scored_for_target.append(row)
            scored = pd.DataFrame(scored_for_target)
            if scored.empty:
                continue
            selected = (
                scored.sort_values(["cal_mae", "avg_cost_seconds", "candidate", "calibration"], kind="mergesort")
                .groupby(["dataset", "target", "seed"], as_index=False, dropna=False)
                .head(1)
            )
            for _idx, row in selected.iterrows():
                pred_table = aggregated[str(row["level"])].reindex(y_all.index)
                x = pred_table[str(row["candidate"])].to_numpy(dtype=np.float64)
                y = y_all.to_numpy(dtype=np.float64)
                valid = np.isfinite(x) & np.isfinite(y)
                train_mask = train_mask_by_mol.reindex(y_all.index).to_numpy() & valid
                calibrated = _calibrated_predictions(str(row["calibration"]), x[train_mask], y[train_mask], x[valid])
                pred_full = np.full_like(y, np.nan, dtype=np.float64)
                pred_full[np.flatnonzero(valid)] = calibrated
                test_mask = test_mask_by_mol.reindex(y_all.index).to_numpy() & valid
                test_metrics = _metrics(y[test_mask], pred_full[test_mask])
                for metric in ("mae", "rmse", "r2"):
                    selected.loc[_idx, metric] = test_metrics[metric]
                pred_rows = pd.DataFrame(
                    {
                        "molecule_id": y_all.index.to_numpy(),
                        "target": target,
                        "seed": seed,
                        "split": split_by_mol.reindex(y_all.index).to_numpy(),
                        "level": row["level"],
                        "candidate": row["candidate"],
                        "calibration": row["calibration"],
                        "true": y,
                        "pred": pred_full,
                    }
                )
                predictions.append(pred_rows[pred_rows["split"].eq("test")])
            selections.append(selected)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(args.out / "axis_formula_metrics.csv", index=False)
    if selections:
        selected_df = pd.concat(selections, ignore_index=True)
        selected_df.to_csv(args.out / "selected_by_cal.csv", index=False)
    else:
        selected_df = pd.DataFrame()
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(args.out / "axis_formula_test_predictions.csv", index=False)

    if not metrics_df.empty:
        mean_std = (
            selected_df.groupby(["dataset", "target", "split_mode", "method", "level", "candidate", "calibration"], dropna=False)
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"),
                r2_mean=("r2", "mean"),
                cal_mae_mean=("cal_mae", "mean"),
                avg_cost_seconds_mean=("avg_cost_seconds", "mean"),
                n_seeds=("seed", "nunique"),
            )
            .reset_index()
            .sort_values("mae_mean")
        )
        mean_std.to_csv(args.out / "selected_by_cal_mean_std.csv", index=False)
        print(mean_std.to_string(index=False))

    write_json(
        args.out / "source_meta.json",
        meta
        | {
            "method": "pseudo-axis analytic Sterimol formula with calibration-selected candidate",
            "feature_cache": str(args.cache),
            "split_mode": args.split_mode,
            "split_fractions": [float(x) for x in split_fractions],
            "axis_mode": args.axis_mode,
            "weight_mode": args.weight_mode,
            "radii": [float(x) for x in args.radii],
            "note": "Formula features use conformer coordinates and Boltzmann weights. Conformer target labels are cached only for diagnostics and are not used for prediction.",
        },
    )


if __name__ == "__main__":
    main()

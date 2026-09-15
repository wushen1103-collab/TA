from __future__ import annotations

import math
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from tqdm import tqdm

from .features import (
    CORE_DESCRIPTOR_NAMES,
    CONF_FUNCS,
    CONF_DESCRIPTOR_NAMES,
    DESC_FUNCS,
    FeatureBundle,
    _all_2d_descriptors,
    _core_descriptors,
    _conf_shape_descriptor_row,
    _fingerprints,
    _safe_float,
)
from .utils import Timer, ensure_dir


KRAKEN_GDRIVE_ID = "1QrV651Re7s6UF7Lg4KC9PM5QQUMPU7wd"
KRAKEN_TARGETS = ["sterimol_B5", "sterimol_L", "sterimol_burB5", "sterimol_burL"]
KRAKEN_LEVELS = ["L2", "L3_4", "L3_8", "L4_32"]


def maybe_download_kraken(zip_path: str | Path) -> Path:
    zip_path = Path(zip_path)
    if zip_path.exists():
        return zip_path
    ensure_dir(zip_path.parent)
    cmd = [sys.executable, "-m", "gdown", KRAKEN_GDRIVE_ID, "-O", str(zip_path)]
    subprocess.run(cmd, check=True)
    return zip_path


def _record_parts(record: Any) -> tuple[str, dict[str, float], dict[str, Any]]:
    if isinstance(record, dict):
        return record["smiles"], record["boltz_avg_properties"], record["conformer_dict"]
    if isinstance(record, (list, tuple)) and len(record) == 3:
        smiles, targets, conformers = record
        return smiles, targets, conformers
    raise TypeError(f"Unsupported Kraken record format: {type(record)!r}")


def load_kraken(zip_path: str | Path, download: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    zip_path = maybe_download_kraken(zip_path) if download else Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Kraken archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("Kraken.pickle") as f:
            raw = pickle.load(f)

    rows = []
    for mol_id, record in raw.items():
        smiles, targets, conformers = _record_parts(record)
        row = {
            "molecule_id": str(mol_id),
            "smiles": smiles,
            "n_source_confs": int(len(conformers)),
        }
        for target in KRAKEN_TARGETS:
            row[target] = _safe_float(targets.get(target, float("nan")), default=float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("molecule_id").reset_index(drop=True)
    meta = {
        "source": "MARCEL Kraken",
        "archive": str(zip_path),
        "n_rows": int(len(df)),
        "targets": KRAKEN_TARGETS,
        "format": "pickle: smiles, boltzmann-average targets, conformer SDFs with weights",
    }
    return df, meta


def _parse_sdf_mol(block: str) -> Chem.Mol | None:
    mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=True)
    if mol is not None and mol.GetNumConformers() > 0:
        return mol
    mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=False)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def _smiles_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid Kraken SMILES: {smiles[:120]!r}")
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        pass
    return mol


def _conf_geom_row(mol: Chem.Mol) -> np.ndarray:
    conf_id = mol.GetConformer().GetId()
    return np.asarray(_conf_shape_descriptor_row(mol, int(conf_id)), dtype=np.float32)


def _weight_entropy(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.0
    w = weights / max(float(weights.sum()), 1e-12)
    return float(-(w * np.log(w + 1e-12)).sum())


def _aggregate_kraken(rows: list[np.ndarray], weights: list[float], k: int) -> np.ndarray:
    width = len(CONF_DESCRIPTOR_NAMES)
    if not rows:
        return np.zeros(width * 6 + 5, dtype=np.float32)

    arr = np.vstack(rows[:k]).astype(np.float32)
    w_raw = np.asarray(weights[:k], dtype=np.float64)
    if not np.isfinite(w_raw).all() or float(w_raw.sum()) <= 0:
        w_raw = np.ones_like(w_raw)
    w = w_raw / float(w_raw.sum())
    weighted_mean = (arr * w[:, None]).sum(axis=0)
    weighted_std = np.sqrt(((arr - weighted_mean) ** 2 * w[:, None]).sum(axis=0))

    parts = [
        arr.mean(axis=0),
        arr.std(axis=0),
        arr.min(axis=0),
        arr.max(axis=0),
        weighted_mean,
        weighted_std,
    ]
    extras = np.asarray(
        [
            len(rows),
            min(k, len(rows)),
            float(np.asarray(weights[:k], dtype=np.float64).sum()),
            float(np.max(weights[:k])) if weights[:k] else 0.0,
            _weight_entropy(w_raw),
        ],
        dtype=np.float32,
    )
    return np.concatenate(parts + [extras]).astype(np.float32)


def _feature_one(record_item: tuple[str, Any]) -> tuple[str, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    mol_id, record = record_item
    smiles, _targets, conformer_dict = _record_parts(record)

    with Timer() as timer_2d:
        mol = _smiles_mol(smiles)
        core = np.asarray(_core_descriptors(mol), dtype=np.float32)
        desc = np.asarray(_all_2d_descriptors(mol), dtype=np.float32)
        morgan, maccs = _fingerprints(mol)

    with Timer() as timer_conf:
        parsed_rows: list[np.ndarray] = []
        parsed_weights: list[float] = []
        for _conf_id, conf_record in conformer_dict.items():
            sdf_block, weight, _conf_targets = conf_record
            sdf_mol = _parse_sdf_mol(sdf_block)
            if sdf_mol is None:
                continue
            parsed_rows.append(_conf_geom_row(sdf_mol))
            parsed_weights.append(_safe_float(weight, default=0.0))

    if parsed_rows:
        order = np.argsort(-np.asarray(parsed_weights, dtype=np.float64))
        parsed_rows = [parsed_rows[int(i)] for i in order]
        parsed_weights = [parsed_weights[int(i)] for i in order]

    l0 = np.concatenate([core, morgan]).astype(np.float32)
    l1 = np.concatenate([l0, desc, maccs]).astype(np.float32)
    conf = {
        "L2": np.concatenate([l1, _aggregate_kraken(parsed_rows, parsed_weights, 1)]).astype(np.float32),
        "L3_4": np.concatenate([l1, _aggregate_kraken(parsed_rows, parsed_weights, 4)]).astype(np.float32),
        "L3_8": np.concatenate([l1, _aggregate_kraken(parsed_rows, parsed_weights, 8)]).astype(np.float32),
        "L4_32": np.concatenate([l1, _aggregate_kraken(parsed_rows, parsed_weights, 32)]).astype(np.float32),
    }

    conf_seconds = float(timer_conf.seconds)
    n_confs = float(len(parsed_rows))
    n_confs_for_time = max(n_confs, 1.0)
    cost = {
        "molecule_id": str(mol_id),
        "cost_L0": 0.0,
        "cost_L1": 0.0,
        "cost_L2": float(min(1.0, n_confs)),
        "cost_L3_4": float(min(4.0, n_confs)),
        "cost_L3_8": float(min(8.0, n_confs)),
        "cost_L4_32": float(min(32.0, n_confs)),
        "wall_seconds_L0": float(timer_2d.seconds),
        "wall_seconds_L1": float(timer_2d.seconds),
        "wall_seconds_L2": float(timer_2d.seconds + conf_seconds * min(1.0, n_confs_for_time) / n_confs_for_time),
        "wall_seconds_L3_4": float(timer_2d.seconds + conf_seconds * min(4.0, n_confs_for_time) / n_confs_for_time),
        "wall_seconds_L3_8": float(timer_2d.seconds + conf_seconds * min(8.0, n_confs_for_time) / n_confs_for_time),
        "wall_seconds_L4_32": float(timer_2d.seconds + conf_seconds * min(32.0, n_confs_for_time) / n_confs_for_time),
        "conf_seconds": conf_seconds,
        "source_n_confs": int(len(conformer_dict)),
        "parsed_n_confs": int(len(parsed_rows)),
        "conf_failed": int(len(parsed_rows) == 0),
        "weight_sum": float(np.sum(parsed_weights)) if parsed_weights else 0.0,
    }
    return str(mol_id), l0, l1, conf, cost


def _kraken_feature_names() -> dict[str, list[str]]:
    level0 = CORE_DESCRIPTOR_NAMES + [f"ecfp4_{i}" for i in range(2048)]
    level1 = level0 + [name for name, _func in DESC_FUNCS] + [f"maccs_{i}" for i in range(167)]
    conf_base = CONF_DESCRIPTOR_NAMES
    conf_names = []
    for stat in ["mean", "std", "min", "max", "wmean", "wstd"]:
        conf_names.extend([f"kraken_{stat}_{name}" for name in conf_base])
    conf_names.extend(["kraken_n_total", "kraken_n_used", "kraken_weight_sum_used", "kraken_weight_max", "kraken_weight_entropy"])
    names = {"L0": level0, "L1": level1}
    for level in KRAKEN_LEVELS:
        names[level] = level1 + conf_names
    return names


def build_kraken_features(
    zip_path: str | Path,
    cache_path: str | Path,
    workers: int = 1,
    force: bool = False,
) -> FeatureBundle:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return joblib.load(cache_path)

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("Kraken.pickle") as f:
            raw = pickle.load(f)
    items = sorted(raw.items(), key=lambda kv: str(kv[0]))

    if workers > 1:
        iterator = Parallel(n_jobs=workers, return_as="generator")(delayed(_feature_one)(item) for item in items)
    else:
        iterator = (_feature_one(item) for item in items)

    l0_rows: list[np.ndarray] = []
    l1_rows: list[np.ndarray] = []
    conf_rows: dict[str, list[np.ndarray]] = {level: [] for level in KRAKEN_LEVELS}
    cost_rows = []

    for _mol_id, l0, l1, conf, cost in tqdm(iterator, total=len(items), desc="kraken-features"):
        if not math.isfinite(float(l0[0])):
            raise RuntimeError("Non-finite feature row encountered")
        l0_rows.append(l0)
        l1_rows.append(l1)
        for level, arr in conf.items():
            conf_rows[level].append(arr)
        cost_rows.append(cost)

    bundle = FeatureBundle(
        arrays={
            "L0": np.vstack(l0_rows),
            "L1": np.vstack(l1_rows),
            **{level: np.vstack(rows) for level, rows in conf_rows.items()},
        },
        costs=pd.DataFrame(cost_rows),
        feature_names=_kraken_feature_names(),
    )
    ensure_dir(cache_path.parent)
    joblib.dump(bundle, cache_path)
    return bundle

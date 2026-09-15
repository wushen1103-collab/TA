from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, Lipinski, MACCSkeys, rdMolDescriptors
from tqdm import tqdm

from .utils import Timer, ensure_dir


CORE_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
]

DESC_FUNCS = [(name, func) for name, func in Descriptors._descList]

CONF_FUNCS = [
    ("Asphericity", Descriptors3D.Asphericity),
    ("Eccentricity", Descriptors3D.Eccentricity),
    ("InertialShapeFactor", Descriptors3D.InertialShapeFactor),
    ("NPR1", Descriptors3D.NPR1),
    ("NPR2", Descriptors3D.NPR2),
    ("PMI1", Descriptors3D.PMI1),
    ("PMI2", Descriptors3D.PMI2),
    ("PMI3", Descriptors3D.PMI3),
    ("RadiusOfGyration", Descriptors3D.RadiusOfGyration),
    ("SpherocityIndex", Descriptors3D.SpherocityIndex),
]

USR_NAMES = [f"USR_{i}" for i in range(12)]
USRCAT_NAMES = [f"USRCAT_{i}" for i in range(60)]
GEOMETRY_EXTRA_NAMES = [
    "LabuteASA",
    "MolVolume",
    "BboxX",
    "BboxY",
    "BboxZ",
    "BboxSpan",
    "BboxVolume",
    "Compactness",
]
CONF_DESCRIPTOR_NAMES = [name for name, _func in CONF_FUNCS] + GEOMETRY_EXTRA_NAMES + USR_NAMES + USRCAT_NAMES

RDLogger.DisableLog("rdApp.warning")


@dataclass
class FeatureBundle:
    arrays: dict[str, np.ndarray]
    costs: pd.DataFrame
    feature_names: dict[str, list[str]]


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        y = float(x)
    except Exception:
        return default
    if not math.isfinite(y):
        return default
    return y


def _core_descriptors(mol: Chem.Mol) -> list[float]:
    return [
        _safe_float(Descriptors.MolWt(mol)),
        _safe_float(Descriptors.MolLogP(mol)),
        _safe_float(rdMolDescriptors.CalcTPSA(mol)),
        _safe_float(Lipinski.NumHAcceptors(mol)),
        _safe_float(Lipinski.NumHDonors(mol)),
        _safe_float(Lipinski.NumRotatableBonds(mol)),
        _safe_float(rdMolDescriptors.CalcNumRings(mol)),
        _safe_float(rdMolDescriptors.CalcFractionCSP3(mol)),
        _safe_float(mol.GetNumHeavyAtoms()),
    ]


def _all_2d_descriptors(mol: Chem.Mol) -> list[float]:
    vals = []
    for _name, func in DESC_FUNCS:
        try:
            vals.append(_safe_float(func(mol)))
        except Exception:
            vals.append(0.0)
    return vals


def _bitvect_to_np(bitvect: DataStructs.cDataStructs.ExplicitBitVect, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def _fingerprints(mol: Chem.Mol, n_bits: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    return _bitvect_to_np(morgan, n_bits), _bitvect_to_np(maccs, 167)


def _conf_descriptor_row(mol: Chem.Mol, conf_id: int, energy: float) -> list[float]:
    vals = _conf_shape_descriptor_row(mol, conf_id)
    vals.append(_safe_float(energy))
    return vals


def _conf_shape_descriptor_row(mol: Chem.Mol, conf_id: int) -> list[float]:
    vals = []
    for _name, func in CONF_FUNCS:
        try:
            vals.append(_safe_float(func(mol, confId=conf_id)))
        except Exception:
            vals.append(0.0)
    coords = None
    try:
        conf = mol.GetConformer(int(conf_id))
        coords = np.asarray(conf.GetPositions(), dtype=np.float64)
    except Exception:
        coords = None
    try:
        labute_asa = _safe_float(rdMolDescriptors.CalcLabuteASA(mol))
    except Exception:
        labute_asa = 0.0
    try:
        mol_volume = _safe_float(AllChem.ComputeMolVolume(mol, confId=int(conf_id)))
    except Exception:
        mol_volume = 0.0
    if coords is None or coords.size == 0:
        bbox = np.zeros(3, dtype=np.float64)
    else:
        bbox = coords.max(axis=0) - coords.min(axis=0)
    bbox_span = float(np.linalg.norm(bbox))
    bbox_volume = float(np.prod(np.maximum(bbox, 1e-6)))
    compactness = mol_volume / bbox_volume if bbox_volume > 0 else 0.0
    vals.extend(
        [
            labute_asa,
            mol_volume,
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            bbox_span,
            bbox_volume,
            compactness,
        ]
    )
    try:
        vals.extend([_safe_float(x) for x in rdMolDescriptors.GetUSR(mol, confId=int(conf_id))])
    except Exception:
        vals.extend([0.0] * len(USR_NAMES))
    try:
        vals.extend([_safe_float(x) for x in rdMolDescriptors.GetUSRCAT(mol, confId=int(conf_id))])
    except Exception:
        vals.extend([0.0] * len(USRCAT_NAMES))
    return vals


def _aggregate(rows: list[list[float]], k: int) -> np.ndarray:
    width = len(CONF_DESCRIPTOR_NAMES) + 1
    if not rows:
        return np.zeros(width * 4 + 2, dtype=np.float32)
    arr = np.asarray(rows[:k], dtype=np.float32)
    parts = [arr.mean(axis=0), arr.std(axis=0), arr.min(axis=0), arr.max(axis=0)]
    extras = np.asarray([len(rows), min(k, len(rows))], dtype=np.float32)
    return np.concatenate(parts + [extras]).astype(np.float32)


def _conformer_features(smiles: str, max_confs: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        width = (len(CONF_DESCRIPTOR_NAMES) + 1) * 4 + 2
        empty = {k: np.zeros(width, dtype=np.float32) for k in ["L2", "L3_4", "L3_8", "L4_32"]}
        return empty, {"conf_seconds": 0.0, "n_confs": 0, "conf_failed": 1}

    mol_h = Chem.AddHs(mol)
    with Timer() as timer:
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = int(seed)
            params.useRandomCoords = True
            params.maxIterations = 1000
            if hasattr(params, "useMacrocycleTorsions"):
                params.useMacrocycleTorsions = True
            if hasattr(params, "useSmallRingTorsions"):
                params.useSmallRingTorsions = True
            params.pruneRmsThresh = 0.5
            params.numThreads = 1
            conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=max_confs, params=params))
            if conf_ids:
                mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
                if mmff_props is not None:
                    results = AllChem.MMFFOptimizeMoleculeConfs(
                        mol_h,
                        numThreads=1,
                        maxIters=200,
                        mmffVariant="MMFF94s",
                    )
                elif AllChem.UFFHasAllMoleculeParams(mol_h):
                    results = AllChem.UFFOptimizeMoleculeConfs(mol_h, numThreads=1, maxIters=200)
                else:
                    results = [(1, 0.0) for _ in conf_ids]
            else:
                results = []
        except Exception:
            conf_ids = []
            results = []
    rows: list[tuple[float, list[float]]] = []
    for pos, conf_id in enumerate(conf_ids):
        energy = float(results[pos][1]) if pos < len(results) and len(results[pos]) > 1 else 0.0
        rows.append((energy, _conf_descriptor_row(mol_h, int(conf_id), energy)))
    rows = sorted(rows, key=lambda item: item[0])
    desc_rows = [r for _energy, r in rows]
    arrays = {
        "L2": _aggregate(desc_rows, 1),
        "L3_4": _aggregate(desc_rows, 4),
        "L3_8": _aggregate(desc_rows, 8),
        "L4_32": _aggregate(desc_rows, 32),
    }
    meta = {"conf_seconds": timer.seconds, "n_confs": len(desc_rows), "conf_failed": int(len(desc_rows) == 0)}
    return arrays, meta


def _featurize_one(smiles: str, max_confs: int, conformer_seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES after cleaning: {smiles}")
    with Timer() as timer_2d:
        core = np.asarray(_core_descriptors(mol), dtype=np.float32)
        desc = np.asarray(_all_2d_descriptors(mol), dtype=np.float32)
        morgan, maccs = _fingerprints(mol)
    conf, meta = _conformer_features(smiles, max_confs=max_confs, seed=conformer_seed)

    l0 = np.concatenate([core, morgan]).astype(np.float32)
    l1 = np.concatenate([l0, desc, maccs]).astype(np.float32)
    conf_rows = {level: np.concatenate([l1, arr]).astype(np.float32) for level, arr in conf.items()}
    conf_seconds = float(meta["conf_seconds"])
    n_confs = max(float(meta["n_confs"]), 1.0)
    cost = {
        "cost_L0": timer_2d.seconds,
        "cost_L1": timer_2d.seconds,
        "cost_L2": timer_2d.seconds + conf_seconds * min(1.0, n_confs) / n_confs,
        "cost_L3_4": timer_2d.seconds + conf_seconds * min(4.0, n_confs) / n_confs,
        "cost_L3_8": timer_2d.seconds + conf_seconds * min(8.0, n_confs) / n_confs,
        "cost_L4_32": timer_2d.seconds + conf_seconds,
        **meta,
    }
    return l0, l1, conf_rows, cost


def build_features(
    df: pd.DataFrame,
    cache_path: str | Path,
    max_confs: int = 32,
    conformer_seed: int = 20260823,
    workers: int = 1,
    force: bool = False,
) -> FeatureBundle:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return joblib.load(cache_path)

    l0_rows: list[np.ndarray] = []
    l1_rows: list[np.ndarray] = []
    conf_rows: dict[str, list[np.ndarray]] = {k: [] for k in ["L2", "L3_4", "L3_8", "L4_32"]}
    cost_rows = []

    smiles_list = df["smiles"].tolist()
    if workers > 1:
        rows = Parallel(n_jobs=workers, return_as="generator")(
            delayed(_featurize_one)(smiles, max_confs, conformer_seed) for smiles in smiles_list
        )
    else:
        rows = (_featurize_one(smiles, max_confs, conformer_seed) for smiles in smiles_list)

    for l0, l1, conf, cost in tqdm(rows, total=len(smiles_list), desc="features"):
        l0_rows.append(l0)
        l1_rows.append(l1)
        for level, arr in conf.items():
            conf_rows[level].append(arr)
        cost_rows.append(cost)

    arrays = {
        "L0": np.vstack(l0_rows),
        "L1": np.vstack(l1_rows),
        **{k: np.vstack(v) for k, v in conf_rows.items()},
    }
    conf_base_names = CONF_DESCRIPTOR_NAMES + ["Energy"]
    conf_names = [f"conf_{stat}_{name}" for stat in ["mean", "std", "min", "max"] for name in conf_base_names]
    conf_names += ["conf_n_total", "conf_n_used"]
    feature_names = {
        "L0": CORE_DESCRIPTOR_NAMES + [f"ecfp4_{i}" for i in range(2048)],
        "L1": CORE_DESCRIPTOR_NAMES
        + [f"ecfp4_{i}" for i in range(2048)]
        + [name for name, _func in DESC_FUNCS]
        + [f"maccs_{i}" for i in range(167)],
    }
    for level in ["L2", "L3_4", "L3_8", "L4_32"]:
        feature_names[level] = feature_names["L1"] + conf_names
    bundle = FeatureBundle(arrays=arrays, costs=pd.DataFrame(cost_rows), feature_names=feature_names)
    ensure_dir(cache_path.parent)
    joblib.dump(bundle, cache_path)
    return bundle

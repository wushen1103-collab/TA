#!/usr/bin/env python3
"""Build leakage-free chemical-distance evidence for the main AD figure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from fcsr.kraken import load_kraken  # noqa: E402


KRAKEN_ZIP = ROOT / "data" / "Kraken.zip"
PREDICTIONS = ROOT / "reference_results" / "analysis" / "fixed_test_predictions.csv"
OUTPUT = HERE / "source_data" / "fig5_ad_residual_distance_source_data.csv"
SUMMARY = HERE / "source_data" / "fig5_ad_domain_summary_source_data.csv"

TARGET_MAP = {
    "sterimol_B5": "B5",
    "sterimol_L": "L",
    "sterimol_burB5": "BurB5",
    "sterimol_burL": "BurL",
}


def fingerprints(smiles: pd.Series) -> dict[str, object]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    output: dict[str, object] = {}
    for molecule_id, smi in smiles.items():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES for molecule {molecule_id}")
        mol = Chem.RemoveHs(mol)
        output[str(molecule_id)] = generator.GetFingerprint(mol)
    return output


def nearest_distance(query_ids: list[str], reference_ids: list[str], fps: dict[str, object]) -> np.ndarray:
    reference = [fps[molecule_id] for molecule_id in reference_ids]
    distance = np.empty(len(query_ids), dtype=float)
    for index, molecule_id in enumerate(query_ids):
        similarity = DataStructs.BulkTanimotoSimilarity(fps[molecule_id], reference)
        distance[index] = 1.0 - float(max(similarity))
    return distance


def reference_loo_distance(reference_ids: list[str], fps: dict[str, object]) -> np.ndarray:
    reference = [fps[molecule_id] for molecule_id in reference_ids]
    distance = np.empty(len(reference_ids), dtype=float)
    for index, fp in enumerate(reference):
        similarity = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, reference), dtype=float)
        similarity[index] = -np.inf
        distance[index] = 1.0 - float(np.max(similarity))
    return distance


def main() -> None:
    molecules, _ = load_kraken(KRAKEN_ZIP, download=False)
    molecules["molecule_id"] = molecules["molecule_id"].astype(str).str.zfill(8)
    molecules = molecules.drop_duplicates("molecule_id").set_index("molecule_id")
    predictions = pd.read_csv(PREDICTIONS, dtype={"molecule_id": str})
    predictions["molecule_id"] = predictions["molecule_id"].str.zfill(8)
    predictions["target_short"] = predictions["target"].map(TARGET_MAP)
    if predictions["target_short"].isna().any():
        raise ValueError("Unknown target in prediction file")

    expected = 5 * 4 * 311
    if len(predictions) != expected:
        raise ValueError(f"Expected {expected} prediction rows, found {len(predictions)}")
    fps = fingerprints(molecules["smiles"])
    all_ids = set(molecules.index)
    distance_rows = []
    for seed in sorted(predictions["seed"].unique()):
        seed_rows = predictions[predictions["seed"] == seed]
        target_test_sets = seed_rows.groupby("target_short")["molecule_id"].apply(set)
        if len({frozenset(ids) for ids in target_test_sets}) != 1:
            raise ValueError(f"Targets do not share a test split for seed {seed}")
        test_ids = sorted(next(iter(target_test_sets)))
        reference_ids = sorted(all_ids - set(test_ids))
        if len(test_ids) != 311 or len(reference_ids) != 1241:
            raise ValueError(f"Unexpected split sizes for seed {seed}")

        test_distance = nearest_distance(test_ids, reference_ids, fps)
        reference_distance = reference_loo_distance(reference_ids, fps)
        threshold = float(np.quantile(reference_distance, 0.95))
        for molecule_id, distance in zip(test_ids, test_distance):
            distance_rows.append(
                {
                    "seed": int(seed),
                    "molecule_id": molecule_id,
                    "tanimoto_dissimilarity": float(distance),
                    "reference_loo_p95": threshold,
                    "domain": "OOD" if distance > threshold else "ID",
                }
            )

    distances = pd.DataFrame(distance_rows)
    distances["distance_decile"] = distances.groupby("seed")["tanimoto_dissimilarity"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 10, labels=False) + 1
    )
    merged = predictions.merge(distances, on=["seed", "molecule_id"], validate="many_to_one")
    reference_mae = merged.groupby("target_short")["abs_error"].mean()
    merged["normalised_abs_error"] = merged["abs_error"] / merged["target_short"].map(reference_mae)
    merged["fingerprint"] = "Morgan radius 2, 2048 bits"
    merged["reference_domain"] = "all non-test molecules for the same outer seed"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)

    summary = (
        merged.groupby(["target_short", "domain", "seed"], as_index=False)
        .agg(
            n_molecules=("molecule_id", "nunique"),
            mae=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            median_dissimilarity=("tanimoto_dissimilarity", "median"),
        )
        .groupby(["target_short", "domain"], as_index=False)
        .agg(
            mean_test_molecules_per_seed=("n_molecules", "mean"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            median_abs_error=("median_abs_error", "mean"),
            median_dissimilarity=("median_dissimilarity", "mean"),
        )
    )
    summary.to_csv(SUMMARY, index=False)

    print(f"Wrote {len(merged)} rows to {OUTPUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

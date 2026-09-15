from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_for_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or smiles


def _stable_group_key(scaffold: str, seed: int) -> str:
    payload = f"{seed}:{scaffold}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def scaffold_split(
    df: pd.DataFrame,
    seed: int = 20260823,
    fractions: tuple[float, float, float, float] = (0.7, 0.1, 0.1, 0.1),
) -> pd.Series:
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1")

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, smiles in enumerate(df["smiles"].tolist()):
        groups[scaffold_for_smiles(smiles)].append(idx)

    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), _stable_group_key(kv[0], seed)))
    n = len(df)
    targets = np.cumsum(np.array(fractions) * n)
    names = np.array(["train", "val", "cal", "test"])

    split = np.empty(n, dtype=object)
    counts = np.zeros(4, dtype=int)
    cursor = 0
    for _scaffold, indices in ordered:
        while cursor < 3 and counts[: cursor + 1].sum() + len(indices) > targets[cursor]:
            cursor += 1
        split[indices] = names[cursor]
        counts[cursor] += len(indices)
    return pd.Series(split, index=df.index, name="split")

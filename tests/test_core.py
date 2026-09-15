import numpy as np
import pandas as pd

from scripts.run_sterimol_axis_formula import _random_split, _safe_unit, _sterimol_features_for_axis


def test_safe_unit() -> None:
    assert np.allclose(_safe_unit(np.array([3.0, 0.0, 0.0])), [1.0, 0.0, 0.0])
    assert np.allclose(_safe_unit(np.zeros(3)), np.zeros(3))


def test_random_split_is_deterministic_and_disjoint() -> None:
    frame = pd.DataFrame({"molecule_id": [str(i) for i in range(100)]})
    first = _random_split(frame, 3, (0.7, 0.0, 0.1, 0.2))
    second = _random_split(frame, 3, (0.7, 0.0, 0.1, 0.2))
    assert first.equals(second)
    assert first.value_counts().to_dict() == {"train": 70, "test": 20, "cal": 10}


def test_sterimol_projection_on_simple_axis() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    radii = np.ones(3)
    result = _sterimol_features_for_axis(
        coords, coords[0], radii, np.ones(3, dtype=bool), np.array([1.0, 0.0, 0.0]), np.array([4.0])
    )
    assert np.isclose(result["Lpos_all"], 3.0)
    assert np.isclose(result["Bmax_all"], 4.0)

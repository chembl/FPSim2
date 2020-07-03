import numpy as np
from FPSim2.FPSim2lib import (
    py_popcount,
    _similarity_search,
    _substructure_search,
    sort_results,
)
from FPSim2 import FPSim2Engine
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
fpe = FPSim2Engine(os.path.join(TESTS_DIR, "data/test.h5"))


def test_py_popcount():
    assert py_popcount(fpe.fps[0]) == 39
    assert py_popcount(fpe.fps[1]) == 45
    assert py_popcount(fpe.fps[2]) == 43


def test_tanimoto_similarity_search():
    res = _similarity_search(fpe.fps[0], fpe.fps, 0.0, 0.0, 0.0, 0, 0, fpe.fps.shape[0])
    np.testing.assert_array_almost_equal(
        res[0:5]["coeff"],
        np.array(
            [
                (0, 4, 1.0),
                (3, 6, 0.63829786),
                (5, 5, 0.625),
                (7, 2, 0.5882353),
                (6, 1, 0.53846157),
            ],
            dtype=[("idx", "<u4"), ("mol_id", "<u4"), ("coeff", "<f4")],
        )["coeff"],
    )
    res = _similarity_search(fpe.fps[0], fpe.fps, 0.6, 0.0, 0.0, 0, 0, fpe.fps.shape[0])
    assert res.shape[0] == 3


def test_tversky_similarity_search():
    res = _similarity_search(fpe.fps[0], fpe.fps, 0.0, 0.5, 0.5, 1, 0, fpe.fps.shape[0])
    np.testing.assert_array_almost_equal(
        res[0:5]["coeff"],
        np.array(
            [
                (0, 4, 1.0),
                (3, 6, 0.779221),
                (5, 5, 0.769231),
                (7, 2, 0.740741),
                (6, 1, 0.7),
            ],
            dtype=[("idx", "<u4"), ("mol_id", "<u4"), ("coeff", "<f4")],
        )["coeff"],
    )
    res = _similarity_search(fpe.fps[0], fpe.fps, 0.6, 0.5, 0.5, 1, 0, fpe.fps.shape[0])
    assert res.shape[0] == 9


def test_substructure_search():
    res = _substructure_search(fpe.fps[0], fpe.fps, 1, 0, 0, 2, 0, fpe.fps.shape[0])
    np.testing.assert_array_equal(res, np.array([4], dtype=np.uint32))


def test_sort_results():
    res = _similarity_search(fpe.fps[0], fpe.fps, 0.6, 0.0, 0.0, 0, 0, fpe.fps.shape[0])
    res.sort(order="coeff")
    sort_results(res)
    np.testing.assert_array_almost_equal(
        res["coeff"],
        np.array(
            [(0, 4, 1.0), (3, 6, 0.638298), (5, 5, 0.625)],
            dtype=[("idx", "<u4"), ("mol_id", "<u4"), ("coeff", "<f4")],
        )["coeff"],
    )

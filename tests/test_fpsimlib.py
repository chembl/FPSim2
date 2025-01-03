import numpy as np
from FPSim2.FPSim2lib.utils import (
    PyPopcount,
    SortResults,
)
from FPSim2.FPSim2lib import (
    GenericSearch,
    TverskySearch,
    SubstructureScreenout,
)
from FPSim2.io.chem import METRICS
from FPSim2 import FPSim2Engine
import os


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
fpe = FPSim2Engine(os.path.join(TESTS_DIR, "data/test.h5"))


def test_PyPopcount():
    assert PyPopcount(fpe.fps[0]) == 39
    assert PyPopcount(fpe.fps[1]) == 45
    assert PyPopcount(fpe.fps[2]) == 43


def test_TanimotoSearch():
    # py_query, py_db, threshold, k, calc_type, start, end
    res = GenericSearch(
        fpe.fps[0], fpe.fps, 0.0, 0, METRICS["tanimoto"], 0, fpe.fps.shape[0]
    )
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
    # py_query, py_db, threshold, k, calc_type, start, end
    res = GenericSearch(
        fpe.fps[0], fpe.fps, 0.6, 0, METRICS["tanimoto"], 0, fpe.fps.shape[0]
    )
    assert res.shape[0] == 3


def test_TanimotoSearchTopK():
    # Test top-k search with k=3 and threshold=0.0
    # py_query, py_db, threshold, k, calc_type, start, end
    res = GenericSearch(
        fpe.fps[0], fpe.fps, 0.0, 3, METRICS["tanimoto"], 0, fpe.fps.shape[0]
    )
    np.testing.assert_array_almost_equal(
        res["coeff"],
        np.array(
            [(0, 4, 1.0), (3, 6, 0.63829786), (5, 5, 0.625)],
            dtype=[("idx", "<u4"), ("mol_id", "<u4"), ("coeff", "<f4")],
        )["coeff"],
    )


def test_CosineSearch():
    # py_query, py_db, threshold, k, calc_type, start, end
    res = GenericSearch(
        fpe.fps[0], fpe.fps, 0.0, 0, METRICS["cosine"], 0, fpe.fps.shape[0]
    )
    np.testing.assert_array_almost_equal(
        res[0:5]["coeff"],
        np.array(
            [
                (0, 4, 1.0),
                (3, 6, 0.782461),
                (5, 5, 0.773309),
                (7, 2, 0.747667),
                (6, 1, 0.707528),
            ],
            dtype=[("idx", "<u4"), ("mol_id", "<u4"), ("coeff", "<f4")],
        )["coeff"],
    )
    # py_query, py_db, threshold, k, calc_type, start, end
    res = GenericSearch(
        fpe.fps[0], fpe.fps, 0.6, 0, METRICS["cosine"], 0, fpe.fps.shape[0]
    )
    assert res.shape[0] == 9


def test_DiceSearch():
    # py_query, py_db, threshold, k, calc_type, start, end
    res = GenericSearch(
        fpe.fps[0], fpe.fps, 0.0, 0, METRICS["dice"], 0, fpe.fps.shape[0]
    )
    np.testing.assert_array_almost_equal(
        res[0:5]["coeff"],
        np.array(
            [
                (0, 4, 1.0),
                (3, 6, 0.7792208),
                (5, 5, 0.7692308),
                (7, 2, 0.7407407),
                (6, 1, 0.7000000),
            ],
            dtype=[("idx", "<u4"), ("mol_id", "<u4"), ("coeff", "<f4")],
        )["coeff"],
    )
    # py_query, py_db, threshold, k, calc_type, start, end
    res = GenericSearch(
        fpe.fps[0], fpe.fps, 0.6, 0, METRICS["dice"], 0, fpe.fps.shape[0]
    )
    assert res.shape[0] == 9


def test_TverskySearch():
    res = TverskySearch(fpe.fps[0], fpe.fps, 0.0, 0.5, 0.5, 0, fpe.fps.shape[0])
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
    res = TverskySearch(fpe.fps[0], fpe.fps, 0.6, 0.5, 0.5, 0, fpe.fps.shape[0])
    assert res.shape[0] == 9


def test_SubstructureScreenout():
    res = SubstructureScreenout(fpe.fps[0], fpe.fps, 0, fpe.fps.shape[0])
    np.testing.assert_array_equal(res, np.array([4], dtype=np.uint32))


def test_SortResults():
    # py_query, py_db, threshold, k, calc_type, start, end
    res = GenericSearch(
        fpe.fps[0], fpe.fps, 0.6, 0, METRICS["tanimoto"], 0, fpe.fps.shape[0]
    )
    res.sort(order="coeff")
    SortResults(res)
    np.testing.assert_array_almost_equal(
        res["coeff"],
        np.array(
            [(0, 4, 1.0), (3, 6, 0.638298), (5, 5, 0.625)],
            dtype=[("idx", "<u4"), ("mol_id", "<u4"), ("coeff", "<f4")],
        )["coeff"],
    )
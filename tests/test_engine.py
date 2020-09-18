from FPSim2 import FPSim2Engine
from rdkit import Chem, DataStructs
import numpy as np
import pytest
import math
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

query_smi = "Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1Cl"
FP_PARAMS = {"radius": 2, "nBits": 2048}

MATRIX = np.matrix(
    [
        [
            0.0,
            0.94285715,
            0.5,
            0.36170214,
            0.5555556,
            0.375,
            0.46153843,
            0.41176468,
            0.4814815,
            0.46296299,
        ],
        [
            0.94285715,
            0.0,
            0.9324324,
            0.9054054,
            0.92105263,
            0.9350649,
            0.9090909,
            0.9240506,
            0.9382716,
            0.9390244,
        ],
        [
            0.5,
            0.9324324,
            0.0,
            0.4528302,
            0.51785713,
            0.33999997,
            0.3333333,
            0.49122804,
            0.29411763,
            0.33962262,
        ],
        [
            0.36170214,
            0.9054054,
            0.4528302,
            0.0,
            0.5344827,
            0.19148934,
            0.26,
            0.24000001,
            0.38181818,
            0.39285713,
        ],
        [
            0.5555556,
            0.92105263,
            0.51785713,
            0.5344827,
            0.0,
            0.4074074,
            0.5084746,
            0.5645161,
            0.5483871,
            0.6,
        ],
        [
            0.375,
            0.9350649,
            0.33999997,
            0.19148934,
            0.4074074,
            0.0,
            0.2745098,
            0.25490195,
            0.3333333,
            0.34545457,
        ],
        [
            0.46153843,
            0.9090909,
            0.3333333,
            0.26,
            0.5084746,
            0.2745098,
            0.0,
            0.375,
            0.26415092,
            0.39655173,
        ],
        [
            0.41176468,
            0.9240506,
            0.49122804,
            0.24000001,
            0.5645161,
            0.25490195,
            0.375,
            0.0,
            0.42372882,
            0.43333334,
        ],
        [
            0.4814815,
            0.9382716,
            0.29411763,
            0.38181818,
            0.5483871,
            0.3333333,
            0.26415092,
            0.42372882,
            0.0,
            0.36206895,
        ],
        [
            0.46296299,
            0.9390244,
            0.33962262,
            0.39285713,
            0.6,
            0.34545457,
            0.39655173,
            0.43333334,
            0.36206895,
            0.0,
        ],
    ],
    dtype=np.float32,
)


def test_validate_against_rdkit():
    in_file_smi = os.path.join(TESTS_DIR, "data/10mols.smi")
    in_file_h5 = os.path.join(TESTS_DIR, "data/test.h5")
    with open(in_file_smi) as f:
        smiles = f.readlines()
    fps = [
        Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smi), **FP_PARAMS
        )
        for smi in smiles
    ]
    query = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(query_smi), **FP_PARAMS
    )
    rdresults = sorted(
        [DataStructs.TanimotoSimilarity(query, fp) for fp in fps], reverse=True
    )

    fpe = FPSim2Engine(in_file_h5, storage_backend="pytables")
    results = fpe.similarity(query_smi, 0.0, n_workers=1)["coeff"]
    for rds, fpss in zip(rdresults, results):
        assert math.isclose(rds, fpss, rel_tol=1e-7)


def test_load_fps():
    in_file = os.path.join(TESTS_DIR, "data/test.h5")
    fpe = FPSim2Engine(in_file, storage_backend="pytables")
    assert fpe.fps.shape[0] == 10
    assert fpe.fps.shape[1] == 34
    assert fpe.popcnt_bins != []


def test_load_fps_sort():
    in_file = os.path.join(TESTS_DIR, "data/test.h5")
    fpe = FPSim2Engine(in_file, storage_backend="pytables")
    fpe2 = FPSim2Engine(in_file, fps_sort=True)
    assert fpe.popcnt_bins == fpe2.popcnt_bins


@pytest.mark.parametrize("n_workers", (1, 2, 4))
def test_similarity(n_workers):
    in_file = os.path.join(TESTS_DIR, "data/test.h5")
    fpe = FPSim2Engine(in_file, storage_backend="pytables")
    results = fpe.similarity(query_smi, 0.7, n_workers=n_workers)
    assert results.shape[0] == 4
    assert list(results[0]) == [1, 1.0]


@pytest.mark.parametrize("n_workers", (1, 2, 4))
def test_tversky(n_workers):
    in_file = os.path.join(TESTS_DIR, "data/test.h5")
    fpe = FPSim2Engine(in_file, storage_backend="pytables")
    results = fpe.tversky(query_smi, 0.85, 0.5, 0.5, n_workers=n_workers)
    res = np.array(
        [(1, 1.0), (6, 0.85057473)],
        dtype={"names": ["mol_id", "coeff"], "formats": ["<u4", "<f4"]},
    )
    np.testing.assert_array_equal(results, res)


@pytest.mark.parametrize("n_workers", (1, 2, 4))
def test_substructure(n_workers):
    in_file = os.path.join(TESTS_DIR, "data/test.h5")
    fpe = FPSim2Engine(in_file, storage_backend="pytables")
    results = fpe.substructure(query_smi, n_workers=n_workers)
    res = np.array(np.array([1], dtype="<u4"))
    np.testing.assert_array_equal(results, res)


@pytest.mark.parametrize("n_workers", (1, 2, 4))
def test_symmetric_matrix(n_workers):
    in_file = os.path.join(TESTS_DIR, "data/test.h5")
    fpe = FPSim2Engine(in_file, storage_backend="pytables")
    csr_matrix = fpe.symmetric_distance_matrix(0.0, n_workers=n_workers)
    np.testing.assert_array_equal(MATRIX, csr_matrix.todense())


@pytest.mark.parametrize("n_workers", (1, 2, 4))
def test_on_disk_similarity(n_workers):
    in_file = os.path.join(TESTS_DIR, "data/test.h5")
    fpe = FPSim2Engine(in_file, in_memory_fps=False, storage_backend="pytables")
    with pytest.raises(Exception):
        fpe.fps
    results = fpe.on_disk_similarity(query_smi, 0.7, n_workers=n_workers)
    r = np.array(
        [(1, 1.0), (6, 0.74), (7, 0.735849), (5, 0.72549)],
        dtype={
            "names": ["mol_id", "coeff"],
            "formats": ["<u4", "<f4"],
            "offsets": [4, 8],
            "itemsize": 12,
        },
    )
    np.testing.assert_array_almost_equal(results["coeff"], r["coeff"])


@pytest.mark.parametrize("n_workers", (1, 2, 4))
def test_on_disk_tversky(n_workers):
    in_file = os.path.join(TESTS_DIR, "data/test.h5")
    fpe = FPSim2Engine(in_file, in_memory_fps=False, storage_backend="pytables")
    with pytest.raises(Exception):
        fpe.fps
    results = fpe.on_disk_tversky(query_smi, 0.85, 0.5, 0.5, n_workers=n_workers)
    res = np.array(
        [(1, 1.0), (6, 0.85057473)],
        dtype={"names": ["mol_id", "coeff"], "formats": ["<u4", "<f4"]},
    )
    np.testing.assert_array_equal(results, res)


@pytest.mark.parametrize("n_workers", (1, 2, 4))
def test_on_disk_substructure(n_workers):
    in_file = os.path.join(TESTS_DIR, "data/test.h5")
    fpe = FPSim2Engine(in_file, in_memory_fps=False, storage_backend="pytables")
    results = fpe.on_disk_substructure(query_smi, n_workers=n_workers)
    res = np.array(np.array([1], dtype="<u4"))
    np.testing.assert_array_equal(results, res)
    with pytest.raises(Exception):
        fpe.fps

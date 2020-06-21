from FPSim2 import FPSim2Engine
from FPSim2.FPSim2lib import py_popcount
from FPSim2.io.chem import rdmol_to_efp
from FPSim2.io.backends.pytables import create_db_file, sort_db_file
from rdkit import Chem, DataStructs
import numpy as np
import math
import unittest
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

query_smi = "Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1Cl"

FP_TYPE = "Morgan"
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


class TestFPSim2(unittest.TestCase):
    def test_a_rdmol_top_efp(self):
        rdmol = Chem.MolFromSmiles("CCC")
        ok = [
            0,
            140737488355328,
            0,
            0,
            33554432,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1073741824,
            0,
            0,
            0,
            0,
            9223372036854775808,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        assert rdmol_to_efp(rdmol, FP_TYPE, FP_PARAMS) == ok

    def test_b_create_db_file(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.smi")
        out_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        create_db_file(in_file, out_file, FP_TYPE, FP_PARAMS)

        fpe = FPSim2Engine(out_file)
        fp_type, fp_params, _ = fpe.storage.read_parameters()
        self.assertEqual(fp_type, FP_TYPE)
        self.assertEqual(fp_params["radius"], FP_PARAMS["radius"])
        self.assertEqual(fp_params["nBits"], FP_PARAMS["nBits"])
        self.assertEqual(fpe.fps.shape[0], 10)

    def test_c_create_db_file_sdf(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.sdf")
        out_file = os.path.join(TESTS_DIR, "data/10mols_sdf.h5")
        create_db_file(in_file, out_file, FP_TYPE, FP_PARAMS, mol_id_prop="mol_id")
        fpe = FPSim2Engine(out_file)
        fp_type, fp_params, _ = fpe.storage.read_parameters()
        self.assertEqual(fp_type, FP_TYPE)
        self.assertEqual(fp_params["radius"], FP_PARAMS["radius"])
        self.assertEqual(fp_params["nBits"], FP_PARAMS["nBits"])
        self.assertEqual(fpe.fps.shape[0], 10)

    def test_d_create_db_file_list(self):
        out_file = os.path.join(TESTS_DIR, "data/10mols_list.h5")
        create_db_file(
            [["CC", 1], ["CCC", 2], ["CCCC", 3]], out_file, FP_TYPE, FP_PARAMS
        )
        fpe = FPSim2Engine(out_file)
        fp_type, fp_params, _ = fpe.storage.read_parameters()
        self.assertEqual(fp_type, FP_TYPE)
        self.assertEqual(fp_params["radius"], FP_PARAMS["radius"])
        self.assertEqual(fp_params["nBits"], FP_PARAMS["nBits"])
        self.assertEqual(fpe.fps.shape[0], 3)

    def test_f_load_fps(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        fpe = FPSim2Engine(in_file)
        self.assertEqual(fpe.fps.shape[0], 10)
        self.assertEqual(fpe.fps.shape[1], 34)
        self.assertTrue(fpe.popcnt_bins != [])

    def test_g_load_fps_sort(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        fpe = FPSim2Engine(in_file)
        fpe2 = FPSim2Engine(in_file, fps_sort=True)
        self.assertEqual(fpe.popcnt_bins, fpe2.popcnt_bins)

    def test_h_search(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        fpe = FPSim2Engine(in_file)
        results = fpe.similarity(query_smi, 0.7, n_workers=1)
        self.assertEqual(results.shape[0], 4)
        self.assertEqual(list(results[0]), [1, 1.0])

    def test_i_validate_against_rdkit(self):
        in_file_smi = os.path.join(TESTS_DIR, "data/10mols.smi")
        in_file_h5 = os.path.join(TESTS_DIR, "data/10mols.h5")
        with open(in_file_smi) as f:
            smiles = f.readlines()
        fps = [
            Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smi), radius=2, nBits=2048
            )
            for smi in smiles
        ]
        query = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(query_smi), radius=2, nBits=2048
        )
        rdresults = sorted(
            [DataStructs.TanimotoSimilarity(query, fp) for fp in fps], reverse=True
        )

        fpe = FPSim2Engine(in_file_h5)
        results = fpe.similarity(query_smi, 0.0, n_workers=1)["coeff"]
        for rds, fpss in zip(rdresults, results):
            self.assertEqual(True, math.isclose(rds, fpss, rel_tol=1e-7))

    def test_j_on_disk_search(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        fpe = FPSim2Engine(in_file, in_memory_fps=False)
        results = fpe.on_disk_similarity(query_smi, 0.7, chunk_size=2, n_workers=2)
        self.assertEqual(results.shape[0], 4)
        self.assertEqual(list(results[0]), [1, 1.0])

    def test_k_py_popcount(self):
        res = py_popcount(
            np.array(
                [
                    0,
                    140737488355328,
                    0,
                    0,
                    33554432,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1073741824,
                    0,
                    0,
                    0,
                    0,
                    9223372036854775808,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=np.uint64,
            )
        )
        assert res == 4

    def test_l_append_fps(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        fpe = FPSim2Engine(in_file, in_memory_fps=False)
        fpe.storage.append_fps([["CC", 11], ["CCC", 12], ["CCCC", 13]])
        fpe = FPSim2Engine(in_file, in_memory_fps=True)
        self.assertEqual(fpe.fps.shape[0], 13)

    def test_m_sort_db_file(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        sort_db_file(in_file)
        fpe = FPSim2Engine(in_file)
        self.assertEqual(fpe.fps[-1][-1], 48)
        self.assertEqual(fpe.fps[0][-1], 2)

    def test_n_delete_fps(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        fpe = FPSim2Engine(in_file, in_memory_fps=False)
        fpe.storage.delete_fps([11, 12, 13])
        sort_db_file(in_file)
        fpe = FPSim2Engine(in_file)
        self.assertEqual(fpe.fps.shape[0], 10)
        self.assertEqual(fpe.fps[-1][-1], 48)
        self.assertEqual(fpe.fps[0][-1], 35)

    def test_o_single_core_matrix(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        fpe = FPSim2Engine(in_file, fps_sort=False)
        csr_matrix = fpe.symmetric_distance_matrix(0.0, n_workers=1)
        np.testing.assert_array_equal(MATRIX, csr_matrix.todense())

    def test_p_multi_core_matrix(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        fpe = FPSim2Engine(in_file, fps_sort=False)
        csr_matrix = fpe.symmetric_distance_matrix(0.0, n_workers=2)
        np.testing.assert_array_equal(MATRIX, csr_matrix.todense())


if __name__ == "__main__":
    unittest.main()

import math
import numpy as np
import tables as tb
from FPSim2 import FPSim2Engine
from FPSim2.FPSim2lib import py_popcount
from FPSim2.io import (
    append_fps,
    create_db_file,
    delete_fps,
    load_fps,
    rdmol_to_efp,
    sort_db_file,
)
from rdkit import Chem, DataStructs
import unittest
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

query_smi = "Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1Cl"

fp_type = "Morgan"
fp_params = {"radius": 2, "nBits": 2048}


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
        assert rdmol_to_efp(rdmol, fp_type, fp_params) == ok

    def test_b_create_db_file(self):
        in_file = os.path.join(TESTS_DIR, 'data/10mols.smi')
        out_file = os.path.join(TESTS_DIR, 'data/10mols.h5')
        create_db_file(in_file, out_file, fp_type, fp_params)
        with tb.open_file(out_file, mode="r") as fp_file:
            config = fp_file.root.config
            assert config[0] == fp_type
            assert config[1]["radius"] == fp_params["radius"]
            assert config[1]["nBits"] == fp_params["nBits"]
            assert fp_file.root.fps.shape[0] == 10

    def test_c_create_db_file_sdf(self):
        in_file = os.path.join(TESTS_DIR, 'data/10mols.sdf')
        out_file = os.path.join(TESTS_DIR, 'data/10mols_sdf.h5')
        create_db_file(
            in_file, out_file, fp_type, fp_params, mol_id_prop="mol_id"
        )
        with tb.open_file(out_file, mode="r") as fp_file:
            config = fp_file.root.config
            assert config[0] == fp_type
            assert config[1]["radius"] == fp_params["radius"]
            assert config[1]["nBits"] == fp_params["nBits"]
            assert fp_file.root.fps.shape[0] == 10

    def test_d_create_db_file_list(self):
        out_file = os.path.join(TESTS_DIR, 'data/10mols_list.h5')
        create_db_file(
            [["CC", 1], ["CCC", 2], ["CCCC", 3]], out_file, fp_type, fp_params
        )
        with tb.open_file(out_file, mode="r") as fp_file:
            config = fp_file.root.config
            assert config[0] == fp_type
            assert config[1]["radius"] == fp_params["radius"]
            assert config[1]["nBits"] == fp_params["nBits"]
            assert fp_file.root.fps.shape[0] == 3

    def test_f_load_fps(self):
        in_file = os.path.join(TESTS_DIR, 'data/10mols.h5')
        fps = load_fps(in_file)
        assert fps.fps.shape[0] == 10
        assert fps.fps.shape[1] == 34
        assert fps.count_ranges != []

    def test_g_load_fps_sort(self):
        in_file = os.path.join(TESTS_DIR, 'data/10mols.h5')
        fps = load_fps(in_file)
        fps2 = load_fps(in_file, sort=True)
        assert fps2.count_ranges == fps.count_ranges

    def test_h_search(self):
        in_file = os.path.join(TESTS_DIR, 'data/10mols.h5')
        fpe = FPSim2Engine(in_file)
        results = fpe.similarity(query_smi, 0.7, n_workers=1)
        assert results.shape[0] == 4
        assert list(results[0]) == [1, 1.0]

    def test_i_validate_against_rdkit(self):
        in_file_smi = os.path.join(TESTS_DIR, 'data/10mols.smi')
        in_file_h5 = os.path.join(TESTS_DIR, 'data/10mols.h5')
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
            assert True == math.isclose(rds, fpss, rel_tol=1e-7)

    def test_j_on_disk_search(self):
        in_file = os.path.join(TESTS_DIR, 'data/10mols.h5')
        fpe = FPSim2Engine(in_file, in_memory_fps=False)
        results = fpe.on_disk_similarity(query_smi, 0.7, chunk_size=100000, n_workers=2)
        assert results.shape[0] == 4
        assert list(results[0]) == [1, 1.0]

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
        in_file = os.path.join(TESTS_DIR, 'data/10mols.h5')
        append_fps(in_file, [["CC", 11], ["CCC", 12], ["CCCC", 13]])
        fps = load_fps(in_file)
        assert fps.fps.shape[0] == 13

    def test_m_sort_db_file(self):
        in_file = os.path.join(TESTS_DIR, 'data/10mols.h5')
        sort_db_file(in_file)
        fps = load_fps(in_file)
        assert fps.fps[-1][-1] == 48
        assert fps.fps[0][-1] == 2  

    def test_n_delete_fps(self):
        in_file = os.path.join(TESTS_DIR, 'data/10mols.h5')
        delete_fps(in_file, [11, 12, 13])
        sort_db_file(in_file)
        fps = load_fps(in_file)
        assert fps.fps.shape[0] == 10
        assert fps.fps[-1][-1] == 48
        assert fps.fps[0][-1] == 35


if __name__ == "__main__":
    unittest.main()

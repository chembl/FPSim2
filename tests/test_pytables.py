from FPSim2.io.backends.pytables import create_db_file, sort_db_file
from FPSim2 import FPSim2Engine
import tables as tb
import pytest
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

FP_TYPE = "Morgan"
FP_PARAMS = {"radius": 2, "fpSize": 2048}


with tb.open_file(os.path.join(TESTS_DIR, "data/test.h5"), mode="r") as fp_file:
    fps = fp_file.root.fps[:]
    num_fields = len(fps[0])
    fps = fps.view("<u8")
    fps = fps.reshape(int(fps.size / num_fields), num_fields)
    popcnt_bins = fp_file.root.config[4]

@pytest.mark.incremental
class TestPytablesBackend:
    def test_create_db_file_smi(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.smi")
        out_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        create_db_file(in_file, out_file, None, FP_TYPE, FP_PARAMS)

        fpe = FPSim2Engine(out_file, storage_backend="pytables")
        fp_type, fp_params, _, _ = fpe.storage.read_parameters()
        assert fp_type == FP_TYPE
        assert fp_params["radius"] == FP_PARAMS["radius"]
        assert fp_params["fpSize"] == FP_PARAMS["fpSize"]
        assert fpe.fps.shape[0] == 10

    def test_append_fps(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        fpe = FPSim2Engine(in_file, in_memory_fps=False, storage_backend="pytables")
        fpe.storage.append_fps([["CC", 11], ["CCC", 12], ["CCCC", 13]], mol_format="smiles")
        fpe = FPSim2Engine(in_file, in_memory_fps=True)
        assert fpe.fps.shape[0] == 13

    def test_sort_db_file(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        sort_db_file(in_file)
        fpe = FPSim2Engine(in_file, storage_backend="pytables")
        assert fpe.fps[0][-1] == 2
        assert fpe.fps[-1][-1] == 48

    def test_n_delete_fps(self):
        in_file = os.path.join(TESTS_DIR, "data/10mols.h5")
        fpe = FPSim2Engine(in_file, in_memory_fps=False, storage_backend="pytables")
        fpe.storage.delete_fps([11, 12, 13])
        sort_db_file(in_file)
        fpe = FPSim2Engine(in_file)
        assert fpe.fps.shape[0] == 10
        assert fpe.fps[0][-1] == 35
        assert fpe.fps[-1][-1] == 48


def test_create_db_file_sdf():
    in_file = os.path.join(TESTS_DIR, "data/10mols.sdf")
    out_file = os.path.join(TESTS_DIR, "data/10mols_sdf.h5")
    create_db_file(in_file, out_file, None, FP_TYPE, FP_PARAMS, mol_id_prop="mol_id")

    fpe = FPSim2Engine(out_file, storage_backend="pytables")
    fp_type, fp_params, _, _ = fpe.storage.read_parameters()
    assert fp_type == FP_TYPE
    assert fp_params["radius"] == FP_PARAMS["radius"]
    assert fp_params["fpSize"] == FP_PARAMS["fpSize"]
    assert fpe.fps.shape[0] == 10


def test_create_db_file_list():
    out_file = os.path.join(TESTS_DIR, "data/10mols_list.h5")
    create_db_file([["CC", 1], ["CCC", 2], ["CCCC", 3]], out_file, 'smiles', FP_TYPE, FP_PARAMS)

    fpe = FPSim2Engine(out_file, storage_backend="pytables")
    fp_type, fp_params, _, _ = fpe.storage.read_parameters()
    assert fp_type == FP_TYPE
    assert fp_params["radius"] == FP_PARAMS["radius"]
    assert fp_params["fpSize"] == FP_PARAMS["fpSize"]
    assert fpe.fps.shape[0] == 3

def test_calc_popcnt_bins():
    in_file = os.path.join(TESTS_DIR, "data/test.h5")
    fpe = FPSim2Engine(
        in_file, in_memory_fps=True, fps_sort=True, storage_backend="pytables"
    )
    assert fpe.popcnt_bins == popcnt_bins

from FPSim2.io.backends.sqla import create_db_table
from FPSim2 import FPSim2Engine
import tables as tb
import pytest
import sys
import os

try:
    import mysqlclient
except:
    pass

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FP_TYPE = "Morgan"
FP_PARAMS = {"radius": 2, "nBits": 2048}
PG_URL = "mysql://root:root@mysql:3306/mysql"


with tb.open_file(os.path.join(TESTS_DIR, "data/test.h5"), mode="r") as fp_file:
    fps = fp_file.root.fps[:]
    num_fields = len(fps[0])
    fps = fps.view("<u8")
    fps = fps.reshape(int(fps.size / num_fields), num_fields)
    popcnt_bins = fp_file.root.config[3]


@pytest.mark.skipif("mysqlclient" not in sys.modules, reason="requires mysqlclient")
def test_create_db_file_smi():
    in_file = os.path.join(TESTS_DIR, "data/10mols.smi")
    create_db_table(in_file, PG_URL, "fpsim2_fp_smi", FP_TYPE, FP_PARAMS)
    fpe = FPSim2Engine(
        conn_url=PG_URL, table_name="fpsim2_fp_smi", storage_backend="sqla"
    )
    fp_type, fp_params, _ = fpe.storage.read_parameters()
    assert fp_type == FP_TYPE
    assert fp_params["radius"] == FP_PARAMS["radius"]
    assert fp_params["nBits"] == FP_PARAMS["nBits"]
    assert fpe.fps.shape[0] == 10
    assert fpe.popcnt_bins == popcnt_bins
    assert fpe.fps.all() == fps.all()


@pytest.mark.skipif("mysqlclient" not in sys.modules, reason="requires mysqlclient")
def test_create_db_file_sdf():
    in_file = os.path.join(TESTS_DIR, "data/10mols.sdf")
    create_db_table(
        in_file, PG_URL, "fpsim2_fp_sdf", FP_TYPE, FP_PARAMS, mol_id_prop="mol_id"
    )
    fpe = FPSim2Engine(
        conn_url=PG_URL, table_name="fpsim2_fp_sdf", storage_backend="sqla"
    )
    fp_type, fp_params, _ = fpe.storage.read_parameters()
    assert fp_type == FP_TYPE
    assert fp_params["radius"] == FP_PARAMS["radius"]
    assert fp_params["nBits"] == FP_PARAMS["nBits"]
    assert fpe.fps.shape[0] == 10
    assert fpe.popcnt_bins == popcnt_bins
    assert fpe.fps.all() == fps.all()

@pytest.mark.skipif("mysqlclient" not in sys.modules, reason="requires mysqlclient")
def test_create_db_file_list():
    create_db_table(
        [["CC", 1], ["CCC", 2], ["CCCC", 3]],
        PG_URL,
        "fpsim2_fp_list",
        FP_TYPE,
        FP_PARAMS,
    )
    fpe = FPSim2Engine(
        conn_url=PG_URL, table_name="fpsim2_fp_list", storage_backend="sqla"
    )
    fp_type, fp_params, _ = fpe.storage.read_parameters()
    assert fp_type == FP_TYPE
    assert fp_params["radius"] == FP_PARAMS["radius"]
    assert fp_params["nBits"] == FP_PARAMS["nBits"]
    assert fpe.fps.shape[0] == 3

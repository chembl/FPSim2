from FPSim2.io.backends.sqla import create_db_table
from FPSim2 import FPSim2Engine
import tables as tb
import pytest
import sys
import os

try:
    import psycopg2, MySQLdb
except:
    pass

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FP_TYPE = "Morgan"
FP_PARAMS = {"radius": 2, "fpSize": 2048}
PG_URL = "postgresql://postgres:postgres@postgres:5432/postgres"
MYSQL_URL = "mysql://root:root@mysql:3306/mysql"

smiles_list = [
    ["Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1Cl", 1],
    ["Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccc(C#N)cc1", 2],
    ["Cc1cc(-n2ncc(=O)[nH]c2=O)cc(C)c1C(O)c1ccc(Cl)cc1", 3],
    ["Cc1ccc(C(=O)c2ccc(-n3ncc(=O)[nH]c3=O)cc2)cc1", 4],
    ["Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccc(Cl)cc1", 5],
    ["Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1", 6],
    ["Cc1cc(Br)ccc1C(=O)c1ccc(-n2ncc(=O)[nH]c2=O)cc1Cl", 7],
    ["O=C(c1ccc(Cl)cc1Cl)c1ccc(-n2ncc(=O)[nH]c2=O)cc1Cl", 8],
    ["CS(=O)(=O)c1ccc(C(=O)c2ccc(-n3ncc(=O)[nH]c3=O)cc2Cl)cc1", 9],
    ["c1cc2cc(c1)-c1cccc(c1)C[n+]1ccc(c3ccccc31)NCCCCCCCCCCNc1cc[n+](c3ccccc13)C2", 10]
]

with tb.open_file(os.path.join(TESTS_DIR, "data/test.h5"), mode="r") as fp_file:
    fps = fp_file.root.fps[:]
    num_fields = len(fps[0])
    fps = fps.view("<u8")
    fps = fps.reshape(int(fps.size / num_fields), num_fields)
    popcnt_bins = fp_file.root.config[4]


@pytest.mark.skipif("psycopg2" not in sys.modules and "MySQLdb" not in sys.modules, reason="requires db drivers")
@pytest.mark.parametrize("db_url", (PG_URL, MYSQL_URL))
def test_create_db_file_smi(db_url):
    in_file = os.path.join(TESTS_DIR, "data/10mols.smi")
    create_db_table(in_file, db_url, "fpsim2_fp_smi", None, FP_TYPE, FP_PARAMS)
    fpe = FPSim2Engine(
        conn_url=db_url, table_name="fpsim2_fp_smi", storage_backend="sqla"
    )
    fp_type, fp_params, _ ,_ = fpe.storage.read_parameters()
    assert fp_type == FP_TYPE
    assert fp_params["radius"] == FP_PARAMS["radius"]
    assert fp_params["fpSize"] == FP_PARAMS["fpSize"]
    assert fpe.fps.shape[0] == 10
    assert fpe.popcnt_bins == popcnt_bins
    assert fpe.fps.all() == fps.all()
    assert fpe.fps.flags['C_CONTIGUOUS']


@pytest.mark.skipif("psycopg2" not in sys.modules and "MySQLdb" not in sys.modules, reason="requires db drivers")
@pytest.mark.parametrize("db_url", (PG_URL, MYSQL_URL))
def test_create_db_file_sdf(db_url):
    in_file = os.path.join(TESTS_DIR, "data/10mols.sdf")
    create_db_table(
        in_file, db_url, "fpsim2_fp_sdf", None, FP_TYPE, FP_PARAMS, mol_id_prop="mol_id"
    )
    fpe = FPSim2Engine(
        conn_url=db_url, table_name="fpsim2_fp_sdf", storage_backend="sqla"
    )
    fp_type, fp_params, _, _ = fpe.storage.read_parameters()
    assert fp_type == FP_TYPE
    assert fp_params["radius"] == FP_PARAMS["radius"]
    assert fp_params["fpSize"] == FP_PARAMS["fpSize"]
    assert fpe.fps.shape[0] == 10
    assert fpe.popcnt_bins == popcnt_bins
    assert fpe.fps.all() == fps.all()
    assert fpe.fps.flags['C_CONTIGUOUS']

@pytest.mark.skipif("psycopg2" not in sys.modules and "MySQLdb" not in sys.modules, reason="requires db drivers")
@pytest.mark.parametrize("db_url", (PG_URL, MYSQL_URL))
def test_create_db_file_list(db_url):
    create_db_table(
        smiles_list,
        db_url,
        "fpsim2_fp_list",
        "smiles",
        FP_TYPE,
        FP_PARAMS,
    )
    fpe = FPSim2Engine(
        conn_url=db_url, table_name="fpsim2_fp_list", storage_backend="sqla"
    )
    fp_type, fp_params, _, _ = fpe.storage.read_parameters()
    assert fp_type == FP_TYPE
    assert fp_params["radius"] == FP_PARAMS["radius"]
    assert fp_params["fpSize"] == FP_PARAMS["fpSize"]
    assert fpe.fps.shape[0] == 10
    assert fpe.popcnt_bins == popcnt_bins
    assert fpe.fps.all() == fps.all()
    assert fpe.fps.flags['C_CONTIGUOUS']

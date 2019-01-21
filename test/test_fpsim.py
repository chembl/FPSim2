import pytest
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from FPSim2 import FPSim2Engine
from FPSim2.FPSim2lib import py_popcount
from rdkit import Chem, DataStructs
from FPSim2.io import *
from rdkit import Chem
import tables as tb
import numpy as np
import math

query_smi = 'Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1Cl'

fp_type = 'Morgan'
fp_params = {'radius': 2, 'nBits': 2048}


def test_rdmol_top_efp():
    rdmol = Chem.MolFromSmiles('CCC')
    ok = [0, 140737488355328, 0, 0, 33554432, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1073741824, 0, 0, 0, 0, 9223372036854775808, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert rdmol_to_efp(rdmol, fp_type, fp_params) == ok


def test_create_db_file():
    create_db_file('test/10mols.smi', 'test/10mols.h5', fp_type, fp_params)
    with tb.open_file('test/10mols.h5', mode='r') as fp_file:
        config = fp_file.root.config
        assert config[0] == fp_type
        assert config[1]['radius'] == fp_params['radius']
        assert config[1]['nBits'] == fp_params['nBits']
        assert fp_file.root.fps.shape[0] == 10


def test_create_db_file_sdf():
    create_db_file('test/10mols.sdf', 'test/10mols_sdf.h5', fp_type, fp_params, mol_id_prop='mol_id')
    with tb.open_file('test/10mols_sdf.h5', mode='r') as fp_file:
        config = fp_file.root.config
        assert config[0] == fp_type
        assert config[1]['radius'] == fp_params['radius']
        assert config[1]['nBits'] == fp_params['nBits']
        assert fp_file.root.fps.shape[0] == 10


def test_create_db_file_list():
    create_db_file([['CC', 1], ['CCC', 2], ['CCCC', 3]], 'test/10mols_list.h5', fp_type, fp_params)
    with tb.open_file('test/10mols_list.h5', mode='r') as fp_file:
        config = fp_file.root.config
        assert config[0] == fp_type
        assert config[1]['radius'] == fp_params['radius']
        assert config[1]['nBits'] == fp_params['nBits']
        assert fp_file.root.fps.shape[0] == 3


def test_create_db_file_sqla():
    engine = create_engine('sqlite:///test/test.db')
    s = Session(engine)
    sql_query = "select mol_string, mol_id from structure"
    resprox = s.execute(sql_query)

    create_db_file(resprox, 'test/10mols_sqla.h5', fp_type, fp_params)
    with tb.open_file('test/10mols_sqla.h5', mode='r') as fp_file:
        config = fp_file.root.config
        assert config[0] == fp_type
        assert config[1]['radius'] == fp_params['radius']
        assert config[1]['nBits'] == fp_params['nBits']
        assert fp_file.root.fps.shape[0] == 10


def test_load_fps():
    fps = load_fps('test/10mols.h5')
    assert fps.fps.shape[0] == 10
    assert fps.fps.shape[1] == 34
    assert fps.count_ranges != []


def test_load_fps_sort():
    fps = load_fps('test/10mols.h5')
    fps2 = load_fps('test/10mols.h5', sort=True)
    assert fps2.count_ranges == fps.count_ranges 


def test_search():
    fpe = FPSim2Engine('test/10mols.h5')
    results = fpe.similarity(query_smi, 0.7, n_workers=1)
    assert results.shape[0] == 4
    assert list(results[0]) == [1, 1.0]


def test_validate_against_rdkit():

    with open('test/10mols.smi') as f:
        smiles = f.readlines()
    fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 
            radius=2, nBits=2048) for smi in smiles]
    query = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(query_smi), 
                radius=2, nBits=2048)
    rdresults = sorted([DataStructs.TanimotoSimilarity(query, fp) for fp in fps], reverse=True)

    fpe = FPSim2Engine('test/10mols.h5')
    results = fpe.similarity(query_smi, 0.0, n_workers=1)['coeff']

    for rds, fpss in zip(rdresults, results):
        assert True == math.isclose(rds, fpss, rel_tol=1e-7)


def test_on_disk_search():
    fpe = FPSim2Engine('test/10mols.h5', in_memory_fps=False)
    results = fpe.similarity(query_smi, 0.7, n_workers=1)
    assert results.shape[0] == 4
    assert list(results[0]) == [1, 1.0]


def test_py_popcount():
    res = py_popcount(np.array([0, 140737488355328, 0, 0, 33554432, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1073741824, 0, 0, 0, 0, 9223372036854775808, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint64))
    assert res == 4


def test_append_fps():
    append_fps('test/10mols.h5', [['CC', 11], ['CCC', 12], ['CCCC', 13]])
    fps = load_fps('test/10mols.h5')
    assert fps.fps.shape[0] == 13


def test_sort_db_file():
    sort_db_file('test/10mols.h5')
    fps = load_fps('test/10mols.h5')
    assert fps.fps[-1][-1] == 48
    assert fps.fps[0][-1] == 2


def test_delete_fps():
    delete_fps('test/10mols.h5', [11, 12, 13])
    sort_db_file('test/10mols.h5')
    fps = load_fps('test/10mols.h5')
    assert fps.fps.shape[0] == 10
    assert fps.fps[-1][-1] == 48
    assert fps.fps[0][-1] == 35

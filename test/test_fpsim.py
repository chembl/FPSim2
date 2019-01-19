import pytest
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from FPSim2 import search, on_disk_search
from FPSim2.FPSim2lib import py_popcount
from FPSim2.io import *
from rdkit import Chem
import tables as tb
import numpy as np


def test_rdmol_top_efp():
    rdmol = Chem.MolFromSmiles('CCC')
    ok = [0, 140737488355328, 0, 0, 33554432, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1073741824, 0, 0, 0, 0, 9223372036854775808, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert rdmol_to_efp(rdmol, 'Morgan', {'radius': 2, 'nBits': 2048}) == ok


def test_create_fp_file():
    fp_type = 'Morgan'
    fp_params = {'radius': 2, 'nBits': 2048}
    create_fp_file('test/10mols.smi', 'test/10mols.h5', fp_type, fp_params)
    with tb.open_file('test/10mols.h5', mode='r') as fp_file:
        config = fp_file.root.config
        assert config[0] == fp_type
        assert config[1]['radius'] == fp_params['radius']
        assert config[1]['nBits'] == fp_params['nBits']
        assert fp_file.root.fps.shape[0] == 10


def test_create_fp_file_sqla():
    fp_type = 'Morgan'
    fp_params = {'radius': 2, 'nBits': 2048}

    engine = create_engine('sqlite:///test/test.db')
    s = Session(engine)
    sql_query = "select mol_string, mol_id from structure"
    resprox = s.execute(sql_query)

    create_fp_file(resprox, 'test/10mols_sqla.h5', fp_type, fp_params)
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
    query = load_query('Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1Cl', 'test/10mols.h5')
    fps = load_fps('test/10mols.h5')
    results = search(query, fps, threshold=0.7, coeff='tanimoto',  n_threads=1)
    assert results.shape[0] == 4
    assert list(results[0]) == [1, 1.0]


def test_on_disk_search():
    results = on_disk_search('Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1Cl', 'test/10mols.h5', threshold=0.7, coeff='tanimoto', n_processes=1)
    assert results.shape[0] == 4
    assert list(results[0]) == [1, 1.0]


def test_py_popcount():
    res = py_popcount(np.array([0, 140737488355328, 0, 0, 33554432, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1073741824, 0, 0, 0, 0, 9223372036854775808, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint64))
    assert res == 4


def test_append_fps():
    append_fps('test/10mols.h5', [['CC', 11], ['CCC', 12], ['CCCC', 13]])
    fps = load_fps('test/10mols.h5')
    assert fps.fps.shape[0] == 13


def test_sort_fp_file():
    sort_fp_file('test/10mols.h5')
    fps = load_fps('test/10mols.h5')
    print(fps)
    assert fps.fps[-1][-1] == 48
    assert fps.fps[0][-1] == 2


def test_delete_fps():
    delete_fps('test/10mols.h5', [11, 12, 13])
    sort_fp_file('test/10mols.h5')
    fps = load_fps('test/10mols.h5')
    print(fps)
    assert fps.fps.shape[0] == 10
    assert fps.fps[-1][-1] == 48
    assert fps.fps[0][-1] == 35

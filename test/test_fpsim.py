import pytest
from FPSim2.io import *
from rdkit import Chem
import tables as tb


def test_rdmol_top_efp():
    rdmol = Chem.MolFromSmiles('CCC')
    ok = [0, 140737488355328, 0, 0, 33554432, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1073741824, 0, 0, 0, 0, 9223372036854775808, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert rdmol_to_efp(rdmol, 'Morgan', {'radius': 2, 'nBits': 2048}) == ok


def test_create_fp_file():
    fp_type = 'Morgan'
    fp_params = {'radius': 2, 'nBits': 2048}
    create_fp_file('10mols.smi', '10mols.h5', fp_type, fp_params)
    with tb.open_file(fp_filename, mode='r') as fp_file:
        config = fp_file.root.config
        assert config[0] == fp_type
        assert config[1]['radius'] == fp_params['radius']
        assert config[1]['nBits'] == fp_params['nBits']
        assert fp_file.root.fps.shape[0] == 10


def test_load_fps():
    fps = load_fps('10mols.h5')
    assert fps.fps.shape[0] == 10
    assert fps.fps.shape[1] == 34
    assert fps.count_ranges != []

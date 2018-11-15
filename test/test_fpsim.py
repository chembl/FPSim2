import pytest
from FPSim2.io import *
from rdkit import Chem


def test_rdmol_top_efp():
    rdmol = Chem.MolFromSmiles('CCC')
    ok = [0, 140737488355328, 0, 0, 33554432, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1073741824, 0, 0, 0, 0, 9223372036854775808, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert rdmol_to_efp(rdmol, 'Morgan', {'radius': 2, 'nBits': 2048}) == ok
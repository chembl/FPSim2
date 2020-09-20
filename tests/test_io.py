from FPSim2.io.chem import (
    smi_mol_supplier,
    sdf_mol_supplier,
    it_mol_supplier,
    get_mol_supplier,
    get_bounds_range,
    build_fp,
)
from rdkit import Chem
import tables as tb
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

with tb.open_file(os.path.join(TESTS_DIR, "data/test.h5"), mode="r") as fp_file:
    fps = fp_file.root.fps[:]
    num_fields = len(fps[0])
    fps = fps.view("<u8")
    fps = fps.reshape(int(fps.size / num_fields), num_fields)
    popcnt_bins = fp_file.root.config[3]

smiles_list = [
    "Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1Cl",
    "Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccc(C#N)cc1",
    "Cc1cc(-n2ncc(=O)[nH]c2=O)cc(C)c1C(O)c1ccc(Cl)cc1",
    "Cc1ccc(C(=O)c2ccc(-n3ncc(=O)[nH]c3=O)cc2)cc1",
    "Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccc(Cl)cc1",
    "Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1",
    "Cc1cc(Br)ccc1C(=O)c1ccc(-n2ncc(=O)[nH]c2=O)cc1Cl",
    "O=C(c1ccc(Cl)cc1Cl)c1ccc(-n2ncc(=O)[nH]c2=O)cc1Cl",
    "CS(=O)(=O)c1ccc(C(=O)c2ccc(-n3ncc(=O)[nH]c3=O)cc2Cl)cc1",
    "c1cc2cc(c1)-c1cccc(c1)C[n+]1ccc(c3ccccc31)NCCCCCCCCCCNc1cc[n+](c3ccccc13)C2",
]


def test_suppliers():
    smi_file = os.path.join(TESTS_DIR, "data/10mols.smi")
    smi_mols = [
        Chem.MolToSmiles(x[1]) for x in smi_mol_supplier(smi_file, gen_ids=False)
    ]
    sdf_file = os.path.join(TESTS_DIR, "data/10mols.sdf")
    sdf_mols = [
        Chem.MolToSmiles(x[1])
        for x in sdf_mol_supplier(sdf_file, gen_ids=False, mol_id_prop="mol_id")
    ]
    sdfgz_file = os.path.join(TESTS_DIR, "data/10mols.sdf.gz")
    sdfgz_mols = [
        Chem.MolToSmiles(x[1])
        for x in sdf_mol_supplier(sdfgz_file, gen_ids=False, mol_id_prop="mol_id")
    ]
    it_mols = [
        Chem.MolToSmiles(x[1]) for x in it_mol_supplier(smiles_list, gen_ids=True)
    ]
    assert smi_mols == sdf_mols == sdfgz_mols == it_mols


def test_get_mol_supplier():
    assert get_mol_supplier("aaa.sdf") == sdf_mol_supplier
    assert get_mol_supplier("aaa.sdf.gz") == sdf_mol_supplier
    assert get_mol_supplier("aaa.smi") == smi_mol_supplier
    assert get_mol_supplier(smiles_list) == it_mol_supplier


def test_get_bounds_range():
    assert get_bounds_range(
        query=fps[0],
        threshold=0.7,
        a=0,
        b=0,
        ranges=popcnt_bins,
        search_type="tanimoto",
    ) == (0, 10)


def test_build_fp():
    rdmol = Chem.MolFromSmiles(smiles_list[0])
    fp = build_fp(rdmol, "Morgan", {"radius": 2, "nBits": 2048}, 42)
    assert fp == (
        42,
        18014398510563328,
        1152991873351024640,
        0,
        4194304,
        4398046511136,
        8798240636960,
        0,
        0,
        18432,
        0,
        9007199254740992,
        512,
        16777216,
        70368745226240,
        16778240,
        562949953421312,
        1610612736,
        9223372174293729280,
        98304,
        4398046511104,
        1152921504606846976,
        17592320327680,
        16777216,
        1,
        0,
        0,
        17593259786272,
        72567767433216,
        1073741832,
        70368744177668,
        0,
        72057594037927936,
        45,
    )

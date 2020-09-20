from typing import Any, Callable, Iterable as IterableType, Dict, List, Tuple, Union
from FPSim2.FPSim2lib.utils import BitStrToIntList, PyPopcount
from collections.abc import Iterable
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from rdkit import Chem
import numpy as np
import re


# SMILES and InChI regexes
SMILES_RE = r"^([A-IK-Za-ik-z0-9@+\-\[\]\(\)\\/=#%:.$]+)$"
INCHI_RE = r"^((InChI=)(.*?)[^J][0-9a-z+\-\(\)\\\/,.?*;]+)$"


FP_FUNCS = {
    "MACCSKeys": rdMolDescriptors.GetMACCSKeysFingerprint,
    "Avalon": pyAvalonTools.GetAvalonFP,
    "Morgan": rdMolDescriptors.GetMorganFingerprintAsBitVect,
    "TopologicalTorsion": rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect,
    "AtomPair": rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect,
    "RDKit": Chem.RDKFingerprint,
    "RDKPatternFingerprint": Chem.PatternFingerprint,
}


FP_FUNC_DEFAULTS = {
    "MACCSKeys": {},
    "Avalon": {
        "nBits": 512,
        "isQuery": False,
        "resetVect": False,
        "bitFlags": 15761407,
    },
    "Morgan": {
        "radius": 2,
        "nBits": 2048,
        "invariants": [],
        "fromAtoms": [],
        "useChirality": False,
        "useBondTypes": True,
        "useFeatures": False,
    },
    "TopologicalTorsion": {
        "nBits": 2048,
        "targetSize": 4,
        "fromAtoms": 0,
        "ignoreAtoms": 0,
        "atomInvariants": 0,
        "includeChirality": False,
    },
    "AtomPair": {
        "nBits": 2048,
        "minLength": 1,
        "maxLength": 30,
        "fromAtoms": 0,
        "ignoreAtoms": 0,
        "atomInvariants": 0,
        "nBitsPerEntry": 4,
        "includeChirality": False,
        "use2D": True,
        "confId": -1,
    },
    "RDKit": {
        "minPath": 1,
        "maxPath": 7,
        "fpSize": 2048,
        "nBitsPerHash": 2,
        "useHs": True,
        "tgtDensity": 0.0,
        "minSize": 128,
        "branchedPaths": True,
        "useBondOrder": True,
        "atomInvariants": 0,
        "fromAtoms": 0,
        "atomBits": None,
        "bitInfo": None,
    },
    "RDKPatternFingerprint": {"fpSize": 2048, "atomCounts": [], "setOnlyBits": None},
}


def rdmol_to_efp(
    rdmol: Chem.Mol, fp_func: str, fp_params: Dict[str, Any]
) -> List[int]:
    fp = FP_FUNCS[fp_func](rdmol, **fp_params)
    return BitStrToIntList(fp.ToBitString())

def build_fp(rdmol, fp_type, fp_params, mol_id):
    efp = rdmol_to_efp(rdmol, fp_type, fp_params)
    popcnt = PyPopcount(np.array(efp, dtype=np.uint64))
    fp = (mol_id, *efp, popcnt)
    return fp

def load_molecule(mol_string: str) -> Chem.Mol:
    """Reads SMILES, molblock or InChI and returns a RDKit mol.

    Parameters
    ----------
    mol_string : str
         SMILES, molblock or InChI.

    Returns
    -------
    mol: ROMol
        RDKit molecule.
    """
    if re.search(INCHI_RE, mol_string, flags=re.IGNORECASE):
        rdmol = Chem.MolFromInchi(mol_string)
    elif re.match(SMILES_RE, mol_string, flags=0):
        rdmol = Chem.MolFromSmiles(mol_string)
    else:
        rdmol = Chem.MolFromMolBlock(mol_string)
    return rdmol


def get_fp_length(fp_type: str, fp_params: Dict[str, Any]) -> int:
    """Returns the FP length given the name of the FP function and it's parameters.

    Parameters
    ----------
    fp_type : str
         Name of the function used to generate the fingerprints.

    fp_params: dict
        Parameters used to generate the fingerprints.

    Returns
    -------
    fp_length: int
        fp length of the fingerprint.
    """
    fp_length = None
    if "nBits" in fp_params.keys():
        fp_length = fp_params["nBits"]
    elif "fpSize" in fp_params.keys():
        fp_length = fp_params["fpSize"]
    if fp_type == "MACCSKeys":
        fp_length = 166
    if not fp_length:
        raise Exception("fingerprint size is not specified")
    return fp_length


def get_bounds_range(
    query: np.ndarray,
    threshold: Union[float, None],
    a: Union[float, None],
    b: Union[float, None],
    ranges: list,
    search_type: str,
) -> Union[Tuple[int, int], Tuple]:
    query_count = query[-1]
    range_to_keep = []

    for count, c_range in ranges:
        if search_type == "tanimoto":
            max_sim = min(query_count, count) / max(query_count, count)
        elif search_type == "tversky":
            max_sim = min(query_count, count) / (
                a * query_count + b * count + (1 - a - b) * min(query_count, count)
            )
        # substructure (simplified tversky with a=1, b=0)
        elif search_type == "substructure":
            max_sim = min(query_count, count) / query_count
        else:
            break
        if max_sim >= threshold:
            range_to_keep.append(c_range)
    if range_to_keep:
        range_to_keep = (range_to_keep[0][0], range_to_keep[len(range_to_keep) - 1][1])
    return tuple(range_to_keep)


def it_mol_supplier(
    iterable: IterableType, gen_ids: bool, **kwargs
) -> IterableType[Tuple[int, Chem.Mol]]:
    """Generator function that reads from iterables.

    Parameters
    ----------
    iterable : iterable
         Python iterable storing molecules.

    gen_ids: bool
        generate ids or not.

    Yields
    -------
    tuple
        int id and rdkit mol.
    """
    for new_mol_id, mol in enumerate(iterable, 1):
        if isinstance(mol, str):
            mol_string = mol
            mol_id = new_mol_id
        else:
            if gen_ids:
                mol_string = mol[0]
                mol_id = new_mol_id
            else:
                try:
                    mol_string = mol[0]
                    mol_id = int(mol[1])
                except ValueError:
                    raise Exception(
                        "FPSim only supports integer ids for molecules, "
                        "cosinder setting gen_ids=True when running "
                        "create_db_file to autogenerate them."
                    )
        rdmol = load_molecule(mol_string)
        if rdmol:
            yield mol_id, rdmol
        else:
            continue


def smi_mol_supplier(
    filename: str, gen_ids: bool, **kwargs
) -> IterableType[Tuple[int, Chem.Mol]]:
    """Generator function that reads from a .smi file.

    Parameters
    ----------
    filename : str
         .smi file name.

    gen_ids: bool
        generate ids or not.

    Yields
    -------
    tuple
        int id and rdkit mol.
    """
    with open(filename, "r") as f:
        for new_mol_id, mol in enumerate(f, 1):
            # if .smi with single smiles column just add the id
            mol = mol.split()
            if len(mol) == 1:
                smiles = mol[0]
                mol_id = new_mol_id
            else:
                if gen_ids:
                    smiles = mol[0]
                    mol_id = new_mol_id
                else:
                    try:
                        smiles = mol[0]
                        mol_id = int(mol[1])
                    except ValueError:
                        raise Exception(
                            "FPSim only supports integer ids for molecules, "
                            "cosinder setting gen_ids=True when running "
                            "create_db_file to autogenerate them."
                        )
            smiles = smiles.strip()
            rdmol = Chem.MolFromSmiles(smiles)
            if rdmol:
                yield mol_id, rdmol
            else:
                continue


def sdf_mol_supplier(
    filename: str, gen_ids: bool, **kwargs
) -> IterableType[Tuple[int, Chem.Mol]]:
    """Generator function that reads from a .sdf file.

    Parameters
    ----------
    filename : str
        .sdf filename.

    gen_ids: bool
        generate ids or not.

    Yields
    -------
    tuple
        int id and rdkit mol.
    """
    if filename.endswith('.gz'):
        import gzip
        gzf = gzip.open(filename)
        suppl = Chem.ForwardSDMolSupplier(gzf)
    else:
        suppl = Chem.ForwardSDMolSupplier(filename)
    for new_mol_id, rdmol in enumerate(suppl, 1):
        if rdmol:
            if gen_ids:
                mol_id = new_mol_id
            else:
                mol_id = rdmol.GetProp(kwargs["mol_id_prop"])
            try:
                int(mol_id)
            except ValueError:
                raise Exception(
                    "FPSim only supports integer ids for molecules, "
                    "cosinder setting gen_ids=True when running "
                    "create_db_file to autogenerate them."
                )
            yield mol_id, rdmol
        else:
            continue


def get_mol_supplier(io_source: Any) -> Union[Callable[..., IterableType[Tuple[int, Chem.Mol]]], None]:
    """Returns a mol supplier depending on the object type and file extension.

    Parameters
    ----------
    mols_source : str or iterable
        .smi or .sdf filename or iterable.

    fps_sort: bool
        Whether if the FPs should be sorted or not.

    Returns
    -------
    callable
        function that will read the molecules from the input.
    """
    supplier = None
    if isinstance(io_source, str):
        split_source = io_source.split(".")
        if split_source[-1] == 'gz':
            input_type = split_source[-2]
        else:
            input_type = split_source[-1]
        if input_type == "smi":
            supplier = smi_mol_supplier
        elif input_type == "sdf":
            supplier = sdf_mol_supplier
    elif isinstance(io_source, Iterable):
        supplier = it_mol_supplier
    else:
        raise Exception("Invalid input")
    return supplier

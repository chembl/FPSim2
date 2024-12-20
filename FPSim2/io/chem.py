from typing import Any, Callable, Iterable as IterableType, Dict, Tuple, Union
from FPSim2.FPSim2lib.utils import BitStrToIntList, PyPopcount
from collections.abc import Iterable
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ExplicitBitVect
from rdkit.Avalon import pyAvalonTools
from rdkit import Chem
import numpy as np
import re


MOLFILE_RE = r" [vV][23]000$"


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
) -> ExplicitBitVect:
    return FP_FUNCS[fp_func](rdmol, **fp_params)


def build_fp(rdmol, fp_type, fp_params, mol_id):
    efp = rdmol_to_efp(rdmol, fp_type, fp_params)
    return process_fp(efp, mol_id)


def process_fp(fp, mol_id):
    fp = BitStrToIntList(fp.ToBitString())
    popcnt = PyPopcount(np.array(fp, dtype=np.uint64))
    return mol_id, *fp, popcnt


def load_molecule(molecule: Any) -> Chem.Mol:
    """Reads SMILES, molblock or InChI and returns a RDKit mol.

    Parameters
    ----------
    molecule : Any
         Chem.Mol, SMILES, molblock or InChI.

    Returns
    -------
    mol: ROMol
        RDKit molecule.
    """
    if isinstance(molecule, Chem.Mol):
        return molecule
    if re.search(MOLFILE_RE, molecule, flags=re.MULTILINE):
        rdmol = Chem.MolFromMolBlock(molecule)
    elif molecule.startswith("InChI="):
        rdmol = Chem.MolFromInchi(molecule)
    else:
        rdmol = Chem.MolFromSmiles(molecule)
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


def minimal_sanitization(mol):
    """
    Performs minimal sanitization of an RDKit molecule object.

    Reference:
        Adapted from https://rdkit.blogspot.com/2016/09/avoiding-unnecessary-work-and.html

    Args:
        mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object to be minimally sanitized

    Returns:
        rdkit.Chem.rdchem.Mol: The input molecule with updated property cache
    """
    # from https://rdkit.blogspot.com/2016/09/avoiding-unnecessary-work-and.html
    mol.UpdatePropertyCache()
    Chem.FastFindRings(mol)
    return mol


def it_mol_supplier(
    iterable: IterableType, **kwargs
) -> IterableType[Tuple[int, Chem.Mol]]:
    """Generator function that reads from iterables.

    Parameters
    ----------
    iterable : iterable
         Python iterable storing tuples of (molecule_string, molecule_id).

    Yields
    -------
    tuple
        int id and rdkit mol.
    """
    mol_funcs = {
        "smiles": Chem.MolFromSmiles,
        "inchi": Chem.MolFromInchi,
        "molfile": Chem.MolFromMolBlock,
    }

    if kwargs["mol_format"].lower() not in mol_funcs:
        raise ValueError("mol_format must be one of: 'smiles', 'inchi', 'molfile'")

    mol_func = mol_funcs[kwargs["mol_format"].lower()]

    for mol, mol_id in iterable:
        try:
            mol_id = int(mol_id)
        except ValueError:
            raise Exception("FPSim2 only supports integer ids for molecules")

        if isinstance(mol, Chem.Mol):
            rdmol = mol
        else:           
            rdmol = mol_func(mol, sanitize=False)
            if rdmol:
                rdmol = minimal_sanitization(rdmol)
        yield mol_id, rdmol


def smi_mol_supplier(filename: str, **kwargs) -> IterableType[Tuple[int, Chem.Mol]]:
    """Generator function that reads from a .smi file.

    Parameters
    ----------
    filename : str
            .smi file name.

    Yields
    -------
    tuple
        int id and rdkit mol.
    """
    with open(filename, "r") as f:
        for line in f:
            mol = line.strip().split()
            if len(mol) < 2:
                continue
            try:
                smiles = mol[0]
                mol_id = int(mol[1])
            except ValueError:
                raise Exception("FPSim2 only supports integer ids for molecules")
            rdmol = Chem.MolFromSmiles(smiles, sanitize=False)
            if rdmol:
                rdmol = minimal_sanitization(rdmol)
                yield mol_id, rdmol


def sdf_mol_supplier(filename: str, **kwargs) -> IterableType[Tuple[int, Chem.Mol]]:
    """Generator function that reads from a .sdf file.

    Parameters
    ----------
    filename : str
        .sdf filename.

    Yields
    -------
    tuple
        int id and rdkit mol.
    """
    if filename.endswith(".gz"):
        import gzip

        gzf = gzip.open(filename)
        suppl = Chem.ForwardSDMolSupplier(gzf, sanitize=False)
    else:
        suppl = Chem.ForwardSDMolSupplier(filename, sanitize=False)

    for rdmol in suppl:
        if rdmol:
            try:
                mol_id = int(rdmol.GetProp(kwargs["mol_id_prop"]))
            except ValueError:
                raise Exception("FPSim2 only supports integer ids for molecules")
            rdmol = minimal_sanitization(rdmol)
            yield mol_id, rdmol


def get_mol_supplier(
    io_source: Any,
) -> Union[Callable[..., IterableType[Tuple[int, Chem.Mol]]], None]:
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
        if split_source[-1] == "gz":
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

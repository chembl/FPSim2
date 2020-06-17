from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from collections.abc import Iterable
from typing import Any, Callable, Iterable as IterableType, Dict, List, Tuple, Union
import numpy as np
import textwrap
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
    rdmol: Chem.Mol, fp_func: str, fp_func_params: Dict[str, Any]
) -> List[int]:
    """Converts rdkit mol in FPSim2 fp format.

    Args:
        rdmol: rdkit mol.
        fp_func: String name of function to generate fps.
        fp_func_params: Parameter dict for fp_func.
    Returns:
        list with ints packing 64 fp bits each.
    """
    fp = FP_FUNCS[fp_func](rdmol, **fp_func_params)
    splited = textwrap.wrap(fp.ToBitString(), 64)
    efp = [int(x, 2) for x in splited]
    return efp


def load_molecule(mol_string: str) -> Chem.Mol:
    """Reads SMILES, molblock or InChi and returns a rdkit mol.

    Args:
        mol_string: SMILES, molblock or InChi.
    Returns:
        rdkit mol object.
    """
    if re.search(INCHI_RE, mol_string, flags=re.IGNORECASE):
        rdmol = Chem.MolFromInchi(mol_string)
    elif re.match(SMILES_RE, mol_string, flags=0):
        rdmol = Chem.MolFromSmiles(mol_string)
    else:
        rdmol = Chem.MolFromMolBlock(mol_string)
    return rdmol


def get_fp_length(fp_func: str, fp_func_params: Dict[str, Any]) -> int:
    """Returns fp length given the name of a function and it's parameters.

    Args:
        fp_func: Name of the function to generate fps.
        fp_func_params: Parameters dict for the function to generate fps.
    Raises:
        Exception: If can't find fp length for a given fp type.
    Returns:
        fp length for the given fp type.
    """
    fp_length = None
    if "nBits" in fp_func_params.keys():
        fp_length = fp_func_params["nBits"]
    elif "fpSize" in fp_func_params.keys():
        fp_length = fp_func_params["fpSize"]
    if fp_func == "MACCSKeys":
        fp_length = 166
    if not fp_length:
        raise Exception("fingerprint size is not specified")
    return fp_length


def get_bounds_range(
    query: np.ndarray,
    threshold: float,
    a: float,
    b: float,
    ranges: list,
    search_type: str,
) -> Union[Tuple[int, int], list]:
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
    return range_to_keep


def it_supplier(
    iterable: IterableType, gen_ids: bool, **kwargs
) -> IterableType[Tuple[int, Chem.Mol]]:
    """Generator function that reads from iterables.

    Args:
        iterable: Python iterable storing molecules.
        gen_ids: flag to generate new ids.
        kwargs: keyword arguments.
    Raises:
        Exception: If tries to use non int values for id.
    Returns:
        Yields next id and rdkit mol tuple.
    """
    for new_mol_id, mol in enumerate(iterable, 1):
        if len(mol) == 1:
            mol_string = mol[0]
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
                mol_string = mol[0]
        rdmol = load_molecule(mol_string)
        if rdmol:
            yield mol_id, rdmol
        else:
            continue


def smi_mol_supplier(
    filename: str, gen_ids: bool, **kwargs
) -> IterableType[Tuple[int, Chem.Mol]]:
    """Generator function that reads .smi files.

    Args:
        filename: input .smi file name.
        gen_ids: flag to generate new ids.
        kwargs: keyword arguments.
    Raises:
        Exception: If tries to use non int values for id.
    Returns:
        Yields next id and rdkit mol tuple.
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
                    smiles = mol[0].strip()
            rdmol = Chem.MolFromSmiles(smiles)
            if rdmol:
                yield mol_id, rdmol
            else:
                continue


def sdf_mol_supplier(
    filename: str, gen_ids: bool, **kwargs
) -> IterableType[Tuple[int, Chem.Mol]]:
    """Generator function that reads .sdf files.

    Args:
        filename: .sdf filename.
        gen_ids: flag to generate new ids.
        kwargs: keyword arguments.
    Raises:
        Exception: If tries to use non int values for id.
    Returns:
        Yields next id and rdkit mol tuple.
    """
    suppl = Chem.SDMolSupplier(filename)
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


def calc_popcnt_bins(fps: Any, fp_length: int, in_memory: bool = False) -> list:
    """Calcs popcount bins.

    Args:
        fps_table: table storing fps.
        fp_length: length of the fp.
        kwargs: keyword arguments.
    Returns:
        list with popcnt ranges.
    """
    popcnt_bins = []
    if in_memory:
        idx = np.unique(fps[:, -1], return_index=True)
        for i, k in enumerate(zip(*idx)):
            if k[0] == idx[0][-1]:
                popcnt_bins.append((k[0], (k[1], fps.shape[0])))
            else:
                popcnt_bins.append((k[0], (k[1], idx[1][int(i + 1)])))
    else:
        for i in range(0, fp_length + 1):
            idx_gen = (row.nrow for row in fps.where("popcnt == {}".format(str(i))))
            try:
                first_id = next(idx_gen)
            except StopIteration:
                continue
            j = first_id
            for j in idx_gen:
                pass
            cnt_idxs = (first_id, j + 1)
            popcnt_bins.append((i, cnt_idxs))
    return popcnt_bins


def get_mol_suplier(io_source: Any) -> Union[Callable[..., IterableType[Tuple[int, Chem.Mol]]], None]:
    """Returns a mol supplier depending on the object type and file extension.

    Args:
        io_source: source of molecules; smi or sdf filenames,
                   or any iterable object.
    Returns:
        molecule supplier generator.
    """
    supplier = None
    if isinstance(io_source, str):
        input_type = io_source.split(".")[-1]
        if input_type == "smi":
            supplier = smi_mol_supplier
        elif input_type == "sdf":
            supplier = sdf_mol_supplier
    elif isinstance(io_source, Iterable):
        supplier = it_supplier
    else:
        raise Exception("No valid input molecules input.")
    return supplier

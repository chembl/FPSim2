import rdkit
from rdkit import Chem
from sqlalchemy.engine.result import ResultProxy
from FPSim2.FPSim2lib import py_popcount
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from collections import namedtuple
import tables as tb
import numpy as np
import textwrap
import re
import os


BATCH_WRITE_SIZE = 10000


# SMILES and InChI regexes
SMILES_RE = r"^([A-IK-Za-ik-z0-9@+\-\[\]\(\)\\/=#%:.$]+)$"
INCHI_RE = r"^((InChI=)(.*?)[^J][0-9a-z+\-\(\)\\\/,.?*;]+)$"


S_INDEXS = {"tanimoto": 0, "substructure": 2}


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


def rdmol_to_efp(rdmol, fp_func, fp_func_params):
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


def load_molecule(mol_string):
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


def load_query(mol_string, fp_filename):
    """Load query molecule from SMILES, molblock or InChi.

    Args:
        query: SMILES, molblock or InChi.
        fp_filename: FP filename to use in the search.
    Returns:
        Numpy array query molecule.
    """
    rdmol = load_molecule(mol_string)
    with tb.open_file(fp_filename, mode="r") as fp_file:
        # retrieve config from fps file
        config = fp_file.root.config
        fp_func = config[0]
        fp_func_params = config[1]
    # generate the efp
    efp = rdmol_to_efp(rdmol, fp_func, fp_func_params)
    efp.append(py_popcount(np.array(efp, dtype=np.uint64)))
    efp.insert(0, 0)
    return np.array(efp, dtype=np.uint64)


def get_fp_length(fp_func, fp_func_params):
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


def get_bounds_range(query, ranges, threshold, coeff):
    query_count = query[-1]
    range_to_keep = []

    for count, c_range in ranges:
        # tanimoto
        if coeff == 0:
            max_sim = min(query_count, count) / max(query_count, count)
        # substructure
        elif coeff == 2:
            max_sim = min(query_count, count) / query_count
        else:
            break
        if max_sim >= threshold:
            range_to_keep.append(c_range)
    if range_to_keep:
        range_to_keep = (range_to_keep[0][0], range_to_keep[len(range_to_keep) - 1][1])
    return range_to_keep


def it_supplier(io_source, gen_ids, **kwargs):
    """Generator function that reads from iterators.

    Args:
        io_source: py list or sqla ResultProxy.
        gen_ids: flag to generate new ids.
        kwargs: keyword arguments.
    Raises:
        Exception: If tries to use non int values for id.
    Returns:
        Yields next id and rdkit mol tuple.
    """
    for new_mol_id, mol in enumerate(io_source, 1):
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
                except Exception as e:
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


def smi_mol_supplier(io_source, gen_ids, **kwargs):
    """Generator function that reads .smi files.

    Args:
        io_source: input .smi file name.
        gen_ids: flag to generate new ids.
        kwargs: keyword arguments.
    Raises:
        Exception: If tries to use non int values for id.
    Returns:
        Yields next id and rdkit mol tuple.
    """
    with open(io_source, "r") as f:
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
                    except Exception as e:
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


def sdf_mol_supplier(io_source, gen_ids, **kwargs):
    """Generator function that reads .sdf files.

    Args:
        io_source: .sdf filename.
        gen_ids: flag to generate new ids.
        kwargs: keyword arguments.
    Raises:
        Exception: If tries to use non int values for id.
    Returns:
        Yields next id and rdkit mol tuple.
    """
    suppl = Chem.SDMolSupplier(io_source)
    for new_mol_id, rdmol in enumerate(suppl, 1):
        if rdmol:
            if gen_ids:
                mol_id = new_mol_id
            else:
                mol_id = rdmol.GetProp(kwargs["mol_id_prop"])
            try:
                int(mol_id)
            except Exception as e:
                raise Exception(
                    "FPSim only supports integer ids for molecules, "
                    "cosinder setting gen_ids=True when running "
                    "create_db_file to autogenerate them."
                )
            yield mol_id, rdmol
        else:
            continue


def calc_count_ranges(fps, fp_length, in_memory=False):
    """Calcs popcount bins.

    Args:
        fps_table: table storing fps.
        fp_length: length of the fp.
        kwargs: keyword arguments.
    Returns:
        list with popcnt ranges.
    """
    count_ranges = []
    if in_memory:
        idx = np.unique(fps["popcnt"], return_index=True)
        for i, k in enumerate(zip(*idx)):
            if k[0] == idx[0][-1]:
                count_ranges.append((k[0], (k[1], fps.shape[0])))
            else:
                count_ranges.append((k[0], (k[1], idx[1][int(i + 1)])))
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
            count_ranges.append((i, cnt_idxs))
    return count_ranges


def get_mol_suplier(io_source):
    """Returns a mol supplier depending on the object type and file extension.

    Args:
        io_source: source of molecules, smi or sdf filenames, 
                   SQLA ResultProxy or python list.
    Returns:
        molecule supplier generator.
    """
    supplier = None
    if isinstance(io_source, list) or isinstance(io_source, ResultProxy):
        supplier = it_supplier
    else:
        input_type = io_source.split(".")[-1]
        if input_type == "smi":
            supplier = smi_mol_supplier
        elif input_type == "sdf":
            supplier = sdf_mol_supplier
    if not supplier:
        raise Exception("No valid input molecules input.")
    return supplier


def create_db_file(
    io_source,
    out_fname,
    fp_func,
    fp_func_params={},
    mol_id_prop="mol_id",
    gen_ids=False,
    sort_by_popcnt=True,
):
    """Creates FPSim2 fingerprints file from .smi, .sdf files, 
    python lists or SQLA queries.

    Args:
        io_source: .smi, .sdf filename, ResultProxy or list.
        out_fname: FPs output filename.
        fp_func: Name of fingerprint function to use to generate fingerprints.
        fp_func_params: Parameters for the fingerprint function.
        mol_id_prop: Name of the .sdf property to read the molecule id.
        gen_ids: Flag to auto-generate ids for the molecules.
        sort_by_popcnt: Flag to sort or not fps by popcnt.
    Returns:
        None.
    """
    # if params dict is empty use defaults
    if not fp_func_params:
        fp_func_params = FP_FUNC_DEFAULTS[fp_func]

    # get mol supplier
    supplier = get_mol_suplier(io_source)

    fp_length = get_fp_length(fp_func, fp_func_params)

    # set compression
    filters = tb.Filters(complib="blosc", complevel=5)

    # set the output file and fps table
    with tb.open_file(out_fname, mode="w") as fp_file:

        class Particle(tb.IsDescription):
            pass

        # hacky...
        columns = {}
        pos = 1
        columns["fp_id"] = tb.Int64Col(pos=pos)
        for i in range(1, int(fp_length / 64) + 1):
            pos += 1
            columns["f" + str(i)] = tb.UInt64Col(pos=pos)
        columns["popcnt"] = tb.Int64Col(pos=pos + 1)
        Particle.columns = columns

        fps_table = fp_file.create_table(
            fp_file.root, "fps", Particle, "Table storing fps", filters=filters
        )

        # set config table; used fp function, parameters and rdkit version
        param_table = fp_file.create_vlarray(
            fp_file.root, "config", atom=tb.ObjectAtom()
        )
        param_table.append(fp_func)
        param_table.append(fp_func_params)
        param_table.append(rdkit.__version__)

        fps = []
        for mol_id, rdmol in supplier(io_source, gen_ids, mol_id_prop=mol_id_prop):
            efp = rdmol_to_efp(rdmol, fp_func, fp_func_params)
            popcnt = py_popcount(np.array(efp, dtype=np.uint64))
            efp.insert(0, mol_id)
            efp.append(popcnt)
            fps.append(tuple(efp))
            if len(fps) == BATCH_WRITE_SIZE:
                fps_table.append(fps)
                fps = []
        # append last batch < 10k
        fps_table.append(fps)

        # create index so table can be sorted
        fps_table.cols.popcnt.create_index(kind="full")

    if sort_by_popcnt:
        sort_db_file(out_fname)


def append_fps(fp_filename, io_source, mol_id_prop="mol_id"):
    """Appends fps to a FP db file.

    Args:
        fp_filename: FP db filename.
        io_source: .smi or .sdf filename, ResultProxy or list.
    Returns:
        None.
    """
    supplier = get_mol_suplier(io_source)

    with tb.open_file(fp_filename, mode="a") as fp_file:
        fp_func = fp_file.root.config[0]
        fp_func_params = fp_file.root.config[1]
        fps_table = fp_file.root.fps
        new_mols = []
        for mol_id, rdmol in supplier(io_source, False, mol_id_prop=mol_id_prop):
            if not rdmol:
                continue
            efp = rdmol_to_efp(rdmol, fp_func, fp_func_params)
            popcnt = py_popcount(np.array(efp, dtype=np.uint64))
            efp.insert(0, mol_id)
            efp.append(popcnt)
            new_mols.append(tuple(efp))
            if len(new_mols) == BATCH_WRITE_SIZE:
                # append last batch < 10k
                fps_table.append(new_mols)
                new_mols = []
        fps_table.append(new_mols)


def delete_fps(fp_filename, ids_list):
    """Delete fps from FP db file.

    Args:
        fp_filename: FP db filename.
        ids_list: ids to delete list.
    Returns:
        None.
    """
    with tb.open_file(fp_filename, mode="a") as fp_file:
        fps_table = fp_file.root.fps
        for fp_id in ids_list:
            to_delete = [
                row.nrow for row in fps_table.where("fp_id == {}".format(str(fp_id)))
            ]
            fps_table.remove_row(to_delete[0])


def sort_db_file(fp_filename):
    """Sorts an existing FP db file.

    Args:
        fp_filename: FP db filename.
    Returns:
        None.
    """
    # rename not sorted filename
    tmp_filename = fp_filename + "_tmp"
    os.rename(fp_filename, tmp_filename)
    filters = tb.Filters(complib="blosc", complevel=5)

    # copy sorted fps and config to a new file
    with tb.open_file(tmp_filename, mode="r") as fp_file:
        with tb.open_file(fp_filename, mode="w") as sorted_fp_file:
            fp_func = fp_file.root.config[0]
            fp_func_params = fp_file.root.config[1]
            fp_length = get_fp_length(fp_func, fp_func_params)

            # create a sorted copy of the fps table
            dst_fps = fp_file.root.fps.copy(
                sorted_fp_file.root,
                "fps",
                filters=filters,
                copyuserattrs=True,
                overwrite=True,
                stats={
                    "groups": 0,
                    "leaves": 0,
                    "links": 0,
                    "bytes": 0,
                    "hardlinks": 0,
                },
                start=None,
                stop=None,
                step=None,
                chunkshape="keep",
                sortby="popcnt",
                check_CSI=True,
                propindexes=True,
            )

            # set config table; used fp function, parameters and rdkit version
            param_table = sorted_fp_file.create_vlarray(
                sorted_fp_file.root, "config", atom=tb.ObjectAtom()
            )
            param_table.append(fp_func)
            param_table.append(fp_func_params)
            param_table.append(rdkit.__version__)

            # update count ranges
            count_ranges = calc_count_ranges(dst_fps, fp_length)
            param_table.append(count_ranges)

    # remove not sorted file
    os.remove(tmp_filename)


def load_fps(fp_filename, sort=False):
    """Loads FP db file into memory.

    Args:
        fp_filename: FPs filename.
        sort: Flag to sort or not after load the fps.
    Returns:
        namedtuple with fps and count ranges.
    """
    with tb.open_file(fp_filename, mode="r") as fp_file:
        fps = fp_file.root.fps[:]
        # files should be sorted but if the file is updated without sorting it
        # can be also in memory sorted
        if sort:
            fps.sort(order="popcnt")
            fp_func = fp_file.root.config[0]
            fp_func_params = fp_file.root.config[1]
            fp_length = get_fp_length(fp_func, fp_func_params)
            count_ranges = calc_count_ranges(fps, fp_length, in_memory=True)
        else:
            count_ranges = fp_file.root.config[3]
    num_fields = len(fps[0])
    fps = fps.view("<u8")
    fps = fps.reshape(int(fps.size / num_fields), num_fields)
    fps_t = namedtuple("fps", "fps count_ranges")
    return fps_t(fps=fps, count_ranges=count_ranges)

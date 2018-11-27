import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from collections import namedtuple
import tables as tb
from FPSim2.FPSim2lib import py_popcount
import numpy as np
import textwrap
import json
import re
import os


BATCH_WRITE_SIZE = 10000


# SMILES and InChI regexes
SMILES_RE = r'^([^J][A-Za-z0-9@+\-\[\]\(\)\\=#$]+)$'
INCHI_RE = r'^((InChI=)?[^J][0-9a-z+\-\(\)\\\/,]+)$'


COEFFS = {
    'tanimoto': 0,
    'substructure': 2
}


FP_FUNCS = {
    'MACCSKeys': rdMolDescriptors.GetMACCSKeysFingerprint,
    'Avalon': pyAvalonTools.GetAvalonFP,
    'Morgan': rdMolDescriptors.GetMorganFingerprintAsBitVect,
    'TopologicalTorsion': rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect,
    'AtomPair': rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect,
    'RDKit': Chem.RDKFingerprint,
    'RDKPatternFingerprint': Chem.PatternFingerprint
}


FP_FUNC_DEFAULTS = {
    'MACCSKeys': {},
    'Avalon': {
        'nBits': 512, 
        'isQuery': False, 
        'resetVect': False, 
        'bitFlags': 15761407
    },
    'Morgan': {
        'radius': 2, 
        'nBits': 2048, 
        'invariants': [], 
        'fromAtoms': [], 
        'useChirality': False, 
        'useBondTypes': True, 
        'useFeatures': False
    },
    'TopologicalTorsion': {
        'nBits': 2048, 
        'targetSize': 4, 
        'fromAtoms': 0, 
        'ignoreAtoms': 0, 
        'atomInvariants': 0, 
        'includeChirality': False
    },
    'AtomPair': {
        'nBits': 2048, 
        'minLength': 1, 
        'maxLength': 30, 
        'fromAtoms': 0, 
        'ignoreAtoms': 0, 
        'atomInvariants': 0, 
        'nBitsPerEntry': 4, 
        'includeChirality': False, 
        'use2D': True, 
        'confId': -1
    },
    'RDKit': {
        'minPath': 1, 
        'maxPath': 7, 
        'fpSize': 2048, 
        'nBitsPerHash': 2, 
        'useHs': True, 
        'tgtDensity': 0.0, 
        'minSize': 128, 
        'branchedPaths': True, 
        'useBondOrder': True, 
        'atomInvariants': 0, 
        'fromAtoms': 0, 
        'atomBits': None, 
        'bitInfo': None
    },
    'RDKPatternFingerprint': {
        'fpSize': 2048, 
        'atomCounts': [], 
        'setOnlyBits': None
    }
}


def rdmol_to_efp(rdmol, fp_func, fp_func_params):
    fp = FP_FUNCS[fp_func](rdmol, **fp_func_params)
    splited = textwrap.wrap(fp.ToBitString(), 64)
    efp = [int(x, 2) for x in splited]
    return efp


def load_query(query, fp_filename):
    # read query molecule
    if re.match(SMILES_RE, query, flags=0):
        rdmol = Chem.MolFromSmiles(query)
    elif re.search(INCHI_RE, query, flags=re.IGNORECASE):
        rdmol = Chem.MolFromInchi(query)
    else:
        rdmol = Chem.MolFromMolBlock(query)
    with tb.open_file(fp_filename, mode='r') as fp_file:
        # retrieve config from fps file
        config = fp_file.root.config
        fp_func = config[0]
        fp_func_params = config[1]
        if rdkit.__version__ != config[2]:
            print('Warning, FPS were created with RDKit {}, using {}'
                .format(config[2], rdkit.__version__))
    # generate the fpe
    efp = rdmol_to_efp(rdmol, fp_func, fp_func_params)
    return np.array(efp, dtype=np.uint64)


def get_fp_length(fp_func, fp_func_params):
    fp_length = None
    if 'nBits' in fp_func_params.keys():
        fp_length = fp_func_params['nBits']
    elif 'fpSize' in fp_func_params.keys():
        fp_length = fp_func_params['fpSize']
    if fp_func == 'MACCSKeys':
        fp_length = 166
    if not fp_length:
        raise Exception('fingerprint size is not specified')
    return fp_length


def smi_mol_supplier(in_fname, gen_ids, **kwargs):
    with open(in_fname, 'r') as f:
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
                        raise Exception('FPSim only supports integer ids for molecules, '
                                        'cosinder setting gen_ids=True when running '
                                        'create_fp_file to autogenerate them.')
                    smiles = mol[0].strip()
            rdmol = Chem.MolFromSmiles(smiles)
            if rdmol:
                yield mol_id, rdmol
            else:
                continue


def sdf_mol_supplier(in_fname, gen_ids, **kwargs):
    suppl = Chem.SDMolSupplier(in_fname)
    for new_mol_id, rdmol in enumerate(suppl, 1):
        if rdmol:
            if gen_ids:
                mol_id = new_mol_id
            else:
                mol_id = rdmol.GetProp(kwargs['mol_id_prop'])
            try:
                int(mol_id)
            except Exception as e:
                raise Exception('FPSim only supports integer ids for molecules, '
                                'cosinder setting gen_ids=True when running '
                                'create_fp_file to autogenerate them.')
            yield mol_id, rdmol
        else:
            continue


def create_fp_file(in_fname, out_fname, fp_func, fp_func_params={}, mol_id_prop='mol_id', gen_ids=False):
    # if params dict is empty use defaults
    if not fp_func_params:
        fp_func_params = FP_FUNC_DEFAULTS[fp_func]

    # select SMI/SDF supplier depending the file extension
    input_type = in_fname.split('.')[-1]
    if input_type == 'smi':
        supplier = smi_mol_supplier
    elif input_type == 'sdf':
        supplier = sdf_mol_supplier
    else:
        raise Exception('No valid input file provided.')

    fp_length = get_fp_length(fp_func, fp_func_params)

    filters = filters = tb.Filters(complib='blosc', complevel=5)

    # set the output file and fps table
    h5file_out = tb.open_file(out_fname + '_tmp', mode='w')

    class Particle(tb.IsDescription):
        pass

    # hacky...
    columns = {}
    pos = 1
    columns['fp_id'] = tb.UInt64Col(pos=pos)
    for i in range(1, int(fp_length / 64) + 1):
        pos += 1
        columns['f'+str(i)] = tb.UInt64Col(pos=pos)
    columns['popcnt'] = tb.Int64Col(pos=pos+1)
    Particle.columns = columns

    fps_table = h5file_out.create_table(h5file_out.root, 
                                        'fps', 
                                        Particle,
                                        'Table storing fps',
                                        filters=filters)

    # set config table; used fp function, parameters and rdkit version
    param_table = h5file_out.create_vlarray(h5file_out.root, 
                                            'config', 
                                            atom=tb.ObjectAtom())
    param_table.append(fp_func)
    param_table.append(fp_func_params)
    param_table.append(rdkit.__version__)

    fps = []
    for mol_id, rdmol in supplier(in_fname, gen_ids, mol_id_prop=mol_id_prop):
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
    fps_table.cols.popcnt.create_index(kind='full')
    h5file_out.close()

    # sort table (copy a file to another because HDF5 does not adjust the size of the store after removal)
    h5file_out = tb.open_file(out_fname + '_tmp', mode='r')
    fp_file = tb.open_file(out_fname, mode='w')

    # create a sorted copy of the fps table
    dst_fps = h5file_out.root.fps.copy(
        fp_file.root, 'fps', filters=filters, copyuserattrs=True, overwrite=True,
        stats={'groups': 0, 'leaves': 0, 'links': 0, 'bytes': 0, 'hardlinks': 0},
        start=None, stop=None, step=None, chunkshape='keep', sortby='popcnt', 
        check_CSI=True, propindexes=True)

    # copy config vlrarray
    dst_config = h5file_out.root.config.copy(
        fp_file.root, 'config', filters=None, copyuserattrs=True, overwrite=True,
        stats={'groups': 0, 'leaves': 0, 'links': 0, 'bytes': 0, 'hardlinks': 0},
        start=None, stop=None, step=None, chunkshape='keep', sortby=None, 
        check_CSI=False, propindexes=False)

    # calc count ranges for baldi optimisation
    count_ranges = []
    for i in range(0, fp_length + 1):
        idx_gen = (row.nrow for row in dst_fps.where("popcnt == {}".format(str(i))))
        try:
            first_id = next(idx_gen)
        except StopIteration:
            continue
        for j in idx_gen:
            pass
        cnt_idxs = (first_id, j + 1)
        count_ranges.append((i, cnt_idxs))

    # add count ranges to vlarray
    dst_config.append(count_ranges)

    h5file_out.close()
    fp_file.close()
    os.remove(out_fname + '_tmp')


# NEW IN NUMPY 1.6
def structured_to_unstructured(arr, dtype=None, copy=False, casting='unsafe'):
    if arr.dtype.names is None:
        raise ValueError('arr must be a structured array')

    fields = _get_fields_and_offsets(arr.dtype)
    names, dts, counts, offsets = zip(*fields)
    n_fields = len(names)

    if dtype is None:
        out_dtype = np.result_type(*[dt.base for dt in dts])
    else:
        out_dtype = dtype

    # Use a series of views and casts to convert to an unstructured array:

    # first view using flattened fields (doesn't work for object arrays)
    # Note: dts may include a shape for subarrays
    flattened_fields = np.dtype({'names': names,
                                 'formats': dts,
                                 'offsets': offsets,
                                 'itemsize': arr.dtype.itemsize})
    arr = arr.view(flattened_fields)

    # next cast to a packed format with all fields converted to new dtype
    packed_fields = np.dtype({'names': names,
                              'formats': [(out_dtype, c) for c in counts]})
    arr = arr.astype(packed_fields, copy=copy, casting=casting)

    # finally is it safe to view the packed fields as the unstructured type
    return arr.view((out_dtype, sum(counts)))


def _get_fields_and_offsets(dt, offset=0):
    """
    Returns a flat list of (name, dtype, count, offset) tuples of all the
    scalar fields in the dtype "dt", including nested fields, in left
    to right order.
    """
    fields = []
    for name in dt.names:
        field = dt.fields[name]
        if field[0].names is None:
            count = 1
            for size in field[0].shape:
                count *= size
            fields.append((name, field[0], count, field[1] + offset))
        else:
            fields.extend(_get_fields_and_offsets(field[0], field[1] + offset))
    return fields


def repack_fields(a, align=False, recurse=False):
    if not isinstance(a, np.dtype):
        dt = repack_fields(a.dtype, align=align, recurse=recurse)
        return a.astype(dt, copy=False)

    if a.names is None:
        return a

    fieldinfo = []
    for name in a.names:
        tup = a.fields[name]
        if recurse:
            fmt = repack_fields(tup[0], align=align, recurse=True)
        else:
            fmt = tup[0]

        if len(tup) == 3:
            name = (tup[2], name)

        fieldinfo.append((name, fmt))

    dt = np.dtype(fieldinfo, align=align)
    return np.dtype((a.type, dt))


def load_fps(fp_filename, copy=False):
    with tb.open_file(fp_filename, mode='r') as fp_file:
        fps = fp_file.root.fps[:]
        count_ranges = fp_file.root.config[3]
    fnames = [x for x in fps.dtype.names[0:-1]]
    # numpy 1.14 and >= 1.16 return a view, not a copy
    popcnt = structured_to_unstructured(fps[['popcnt']])
    fps2 = structured_to_unstructured(fps[fnames])
    fps_t = namedtuple('fps', 'fps popcnt count_ranges')
    return fps_t(fps=fps2, popcnt=popcnt, count_ranges=count_ranges)

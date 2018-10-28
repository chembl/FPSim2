import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from .thread_safe_tables import tables
from FPSim2.FPSim2lib import py_popcount
import numpy as np
import textwrap
import json
import re


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
        'radius': 3, 
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
    with tables.open_file(fp_filename, mode='r') as fp_file:
        # retrieve config from fps file
        config = fp_file.root.config
        fp_func = config[0]
        fp_func_params = config[1]
        if rdkit.__version__ != config[2]:
            print('Warning, FPS were created with RDKit {}, using {}'
                .format(config[2], rdkit.__version__))
    # generate the fpe
    efp = rdmol_to_efp(rdmol, fp_func, fp_func_params)
    return np.array([efp], dtype=np.uint64)


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
                    mol_id = new_mol_id
                else:
                    try:
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


def create_fp_file(in_fname, out_fname, fp_func, fp_func_params={}, mol_id_prop='mol_id', compress=True, gen_ids=False):
    # if params dict is empty use defaults
    if not fp_func_params:
        fp_func_params = FP_FUNC_DEFAULTS[fp_func]

    # RDKit Morgan FP has no default value for radius
    if fp_func == 'Morgan' and 'radius' not in fp_func_params.keys():
        fp_func_params.update({'radius': 3})

    # select SMI/SDF supplier depending the file extension
    input_type = in_fname.split('.')[-1]
    if input_type == 'smi':
        supplier = smi_mol_supplier
    elif input_type == 'sdf':
        supplier = sdf_mol_supplier
    else:
        raise Exception('No valid input file provided.')

    fp_length = get_fp_length(fp_func, fp_func_params)

    filters = None
    if compress:
        filters = tables.Filters(complib='zlib', complevel=5)

    # set the output file and fps table
    h5file_out = tables.open_file(out_fname, mode='w')
    fps_atom = tables.Atom.from_dtype(np.dtype('uint64'))
    fps_table = h5file_out.create_earray(h5file_out.root,
                                        'fps',
                                        fps_atom,
                                        shape=((0, fp_length / 64 + 2)),
                                        filters=filters)

    # set config table; used fp function, parameters and rdkit version
    param_table = h5file_out.create_vlarray(h5file_out.root, 
                                            'config', 
                                            atom=tables.ObjectAtom())
    param_table.append(fp_func)
    param_table.append(fp_func_params)
    param_table.append(rdkit.__version__)

    fps = []
    for mol_id, rdmol in supplier(in_fname, gen_ids, mol_id_prop=mol_id_prop):
        efp = rdmol_to_efp(rdmol, fp_func, fp_func_params)
        popcnt = py_popcount(np.array([efp], dtype=np.uint64))
        efp.insert(0, mol_id)
        efp.append(popcnt)
        efp = np.asarray(efp, dtype=np.uint64)
        fps.append(efp)
        # insert in batches of 10k fps
        if len(fps) == BATCH_WRITE_SIZE:
            fps_table.append(np.asarray(fps))
            fps = []
    # append last batch < 10k
    fps_table.append(np.asarray(fps))
    h5file_out.close()


def load_fps(fp_filename):
    with tables.open_file(fp_filename, mode='r') as fp_file:
        fps = fp_file.root.fps[:]
    # sort by counts
    # ugly but the only way to do inplace sort in not structured arrays
    fps.view(','.join(['uint64']*fps.shape[1])).sort(order=['f{}'.format(fps.shape[1] - 1)], axis=0)
    idx = np.unique(fps[:,-1], return_index=True)
    return [fps, idx]


def get_disk_memory_size(fp_filename):
    with tables.open_file(fp_filename, mode='r') as fp_file:
        disk = fp_file.root.fps.size_on_disk
        memory = fp_file.root.fps.size_in_memory
    return disk, memory

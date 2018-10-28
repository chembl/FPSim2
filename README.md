# FPSim2: Simple package for fast molecular similarity searches

This is not even a alpha version. Everything is changing and anything that can crash... will eventually do it.
FPSim2 is designed to run fast compound similarity searches and to be easily integrated with any Python web framework in order to expose similarity services.

## Installation 

    python setup.py install

Planning to include it in a conda repo.

### Requirements

FPSim2 is heavily coupled to RDKit. As the easiest way to install it is using conda environments, the use of conda is highly recommended.

* RDKit (install it via conda using rdkit or conda-forge channels)
    - conda install -c conda-forge rdkit
    - conda install -c rdkit rdkit
* PyTables
* Cython


## Usage

### Create FP file

    from FPSim2 import create_fp_file

    # input file, output file, FP type, FP parameters
    create_fp_file('chembl.smi', 'chembl.h5', 'Morgan', {'radius': 2, 'nBits': 2048})

FPSim2 will use RDKit default parameters for a fingerprint type in case no parameters are used. Available FP types and default parameters listed below.

### FP Types

All fingerprints are calculated using RDKit.  

- [MACCSKeys](https://rdkit.org/docs/api/rdkit.Chem.rdMolDescriptors-module.html#GetMACCSKeysFingerprint)
- [Avalon](http://www.rdkit.org/Python_Docs/rdkit.Avalon.pyAvalonTools-module.html#GetAvalonFP)
- [Morgan](https://rdkit.org/docs/api/rdkit.Chem.rdMolDescriptors-module.html#GetMorganFingerprintAsBitVect)
- [TopologicalTorsion](https://rdkit.org/docs/api/rdkit.Chem.rdMolDescriptors-module.html#GetHashedTopologicalTorsionFingerprintAsBitVect)
- [AtomPair](https://rdkit.org/docs/api/rdkit.Chem.rdMolDescriptors-module.html#GetHashedAtomPairFingerprintAsBitVect)
- [RDKit](http://rdkit.org/Python_Docs/rdkit.Chem.rdmolops-module.html#RDKFingerprint)
- [RDKPatternFingerprint](http://rdkit.org/Python_Docs/rdkit.Chem.rdmolops-module.html#PatternFingerprint)


### Limitations

Due to it's simplicity FPSim2 can only use integer ids for FPs, however it can generate new ids for the provided molecules using gen_ids flag.

    create_fp_file('chembl.smi', 'chembl.h5', 'Morgan', {'radius': 2, 'nBits': 2048}, gen_ids=True)

In case RDKit is not able to load a molecule, the id assigned to the molecule will be also skipped so the nth molecule in the input file will have id=n.


### Run a search

    from FPSim2 import run_search

    results = run_search('CC(=O)Oc1ccccc1C(=O)O', 'chembl.h5', threshold=0.7, coeff='tanimoto')
    for r in results:
        print(r)

### Run a in memory search

If your data fits in RAM, you can preload all the fps in memory and run much faster queries.

    from FPSim2 import run_in_memory_search
    from FPSim2.io import load_query, load_fps

    fp_filename = 'chembl.h5'

    query = load_query('CC(=O)Oc1ccccc1C(=O)O', fp_filename)
    fps = load_fps(fp_filename)

    results = run_in_memory_search(query, fps, threshold=0.7, coeff='tanimoto')
    for r in results:
        print(r)

If you want to know if your dataset will fit in memory without loading a single byte:

    from FPSim2.io import get_disk_memory_size

    fp_filename = 'chembl.h5'
    disk, memory = get_disk_memory_size(fp_filename)
    print('Your FPs will need {} MB of available memory'.format(memory / 1024 / 1024))

### Available coefficients

- tanimoto (aka [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index))
- substructure ([Tversky](https://en.wikipedia.org/wiki/Tversky_index) with α=1, β=0). Use it with RDKPatternFingerprint.

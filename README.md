# FPSim2: Simple package for fast molecular similarity searches

FPSim2 is designed to run fast compound similarity searches and to be easily integrated with any Python web framework in order to expose similarity services.

## Installation 

Use a conda environment to install it:

    conda install -c efelix fpsim2 

### Requirements

FPSim2 is heavily coupled to RDKit. Install it via rdkit or conda-forge channels

- conda install -c rdkit rdkit

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

### Run a in memory search

    from FPSim2 import run_in_memory_search
    from FPSim2.io import load_query, load_fps

    fp_filename = 'chembl.h5'

    query = load_query('CC(=O)Oc1ccccc1C(=O)O', fp_filename)
    fps = load_fps(fp_filename)

    results = run_in_memory_search(query, fps, threshold=0.7, coeff='tanimoto')
    for r in results:
        print(r)

### Available coefficients

- tanimoto (aka [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index))
- substructure ([Tversky](https://en.wikipedia.org/wiki/Tversky_index) with α=1, β=0). Use it altogether with RDKPatternFingerprint.

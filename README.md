# FPSim2: Simple package for fast molecular similarity searches

FPSim2 is designed to run fast compound similarity searches with big datasets and to be easily integrated with any Python web framework in order to expose similarity search services. FPSim2 works better using high search thresholds (>=0.7).

Implementing: 

- Using fast population count algorithm(builtin-popcnt-unrolled) using SIMD instructions from https://github.com/WojciechMula/sse-popcount.
- Bounds for sublinear speedups from https://pubs.acs.org/doi/abs/10.1021/ci600358f
- A compressed file format with optimised read speed based in [PyTables](https://www.pytables.org/) and [BLOSC](http://www.blosc.org/pages/blosc-in-depth/).


## Installation 

Use a conda environment to install it. Builds for linux and mac currently available:

    conda install -c efelix fpsim2 

### Requirements

FPSim2 is heavily coupled to RDKit. Install it via rdkit or conda-forge channels

    conda install -c rdkit rdkit
    conda install -c conda-forge rdkit

## Usage

### Create FP file

    from FPSim2 import create_fp_file

    # input file, output file, FP type, FP parameters
    create_fp_file('chembl.smi', 'chembl.h5', 'Morgan', {'radius': 2, 'nBits': 2048})

FPSim2 will use RDKit default parameters for a fingerprint type in case no parameters are used. Available FP types and default parameters listed below.

### FP Types

All fingerprints are calculated using RDKit.  

- [MACCSKeys](http://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint)
- [Avalon](http://rdkit.org/docs/source/rdkit.Avalon.pyAvalonTools.html#rdkit.Avalon.pyAvalonTools.GetAvalonFP)
- [Morgan](http://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect)
- [TopologicalTorsion](http://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect)
- [AtomPair](http://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect)
- [RDKit](http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.RDKFingerprint)
- [RDKPatternFingerprint](http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.PatternFingerprint)


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

    results = run_in_memory_search(query, fps, threshold=0.7, coeff='tanimoto', n_threads=1)

As GIL is most of the time released, searches can be speeded up using multiple threads. This is specially useful when dealing with huge datasets and demanding real time results. Performance will vary depending on the population count distribution of the dataset, the query molecule, the threshold, the number of results and the number of threads used.

### Run a on disk search

If you're searching against a huge dataset or you have small RAM, you can still run searches.

    from FPSim2 import run_search

    fp_filename = 'chembl.h5'
    query_string = 'CC(=O)Oc1ccccc1C(=O)O'

    results = run_search(query_string, fp_filename, threshold=0.7, coeff='tanimoto', chunk_size=1000000, n_processes=1)

In the on disk search variant, parallelisation is achieved with processes. Performance will vary depending on the population count distribution of the dataset, the query molecule, the threshold, the number of results, the chunk size and the number of processes used.

### Available coefficients

- tanimoto (aka [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index))
- substructure ([Tversky](https://en.wikipedia.org/wiki/Tversky_index) with α=1, β=0). Use it altogether with RDKPatternFingerprint.

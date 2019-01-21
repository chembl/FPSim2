[![Build Status](https://travis-ci.org/chembl/FPSim2.svg?branch=master)](https://travis-ci.org/chembl/FPSim2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Anaconda-Server Badge](https://anaconda.org/efelix/fpsim2/badges/version.svg)](https://anaconda.org/efelix/fpsim2)
[![Binder](http://mybinder.org/badge.svg)](http://beta.mybinder.org/v2/gh/eloyfelix/fpsim2_binder/master?filepath=demo.ipynb)


# FPSim2: Simple package for fast molecular similarity searches

FPSim2 is designed to run fast compound similarity searches with big datasets and to be easily integrated with any Python web framework in order to expose similarity search services. FPSim2 works better using high search thresholds (>=0.7).

Implementing: 

- Using fast population count algorithm(builtin-popcnt-unrolled) from https://github.com/WojciechMula/sse-popcount using SIMD instructions.
- Bounds for sublinear speedups from https://pubs.acs.org/doi/abs/10.1021/ci600358f
- A compressed file format with optimised read speed based in [PyTables](https://www.pytables.org/) and [BLOSC](http://www.blosc.org/pages/blosc-in-depth/)
- In memory and on disk search modes


## Installation 

Use a conda environment to install it. Builds available for:
- linux:
    - Python 3.6
    - Python 3.7
- mac:
    - Python 3.6

Python 3.7 for mac build will also be available once [RDKit #2207 issue](https://github.com/rdkit/rdkit/issues/2207) is solved

```
conda install fpsim2 -c efelix -c conda-forge
```

## Usage

### Create FP file

```python
from FPSim2 import create_db_file

# from .smi file
create_db_file('chembl.smi', 'chembl.h5', 'Morgan', {'radius': 2, 'nBits': 2048})

# from .sdf file, need to specify sdf property containing the molecule id
create_db_file('chembl.sdf', 'chembl.h5', 'Morgan', {'radius': 2, 'nBits': 2048}, mol_id_prop='mol_id')

# from Python list
create_db_file([['CC', 1], ['CCC', 2], ['CCCC', 3]], 'test/10mols.h5', 'Morgan', {'radius': 2, 'nBits': 2048})

# from sqlalchemy ResulProxy
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

engine = create_engine('sqlite:///test/test.db')
s = Session(engine)
sql_query = "select mol_string, mol_id from structure"
res_prox = s.execute(sql_query)

create_db_file(res_prox, 'test/10mols.h5', 'Morgan', {'radius': 2, 'nBits': 2048})
```

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

```python
create_db_file('chembl.smi', 'chembl.h5', 'Morgan', {'radius': 2, 'nBits': 2048}, gen_ids=True)
```

In case RDKit is not able to load a molecule, the id assigned to the molecule will be also skipped so the nth molecule in the input file will have id=n.

### Run a in memory search

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_24.h5'
query = 'CC(=O)Oc1ccccc1C(=O)O'

fpe = FPSim2Engine(fp_filename)

results = fpe.similarity(aspirin, 0.7, n_workers=1)
```

As GIL is most of the time released, searches can be speeded up using multiple threads. This is specially useful when dealing with huge datasets and demanding real time results. Performance will vary depending on the population count distribution of the dataset, the query molecule, the threshold, the number of results and the number of threads used.

### Run a on disk search

If you're searching against a huge dataset or you have small RAM, you can still run searches.

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_24.h5'
query = 'CC(=O)Oc1ccccc1C(=O)O'

fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

results = fpe.on_disk_similarity(aspirin, 0.7, chunk_size=250000, n_workers=1)
```

In the on disk search variant, parallelisation is achieved with processes. Performance will vary depending on the population count distribution of the dataset, the query molecule, the threshold, the number of results, the chunk size and the number of processes used.

### Available coefficients

- tanimoto (aka [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index))
- substructure ([Tversky](https://en.wikipedia.org/wiki/Tversky_index) with α=1, β=0). Use it altogether with RDKPatternFingerprint.

## Trying it online

To try out FPSim2 interactively in your web browser, just click on the binder [![Binder](http://mybinder.org/badge.svg)](http://beta.mybinder.org/v2/gh/eloyfelix/fpsim2_binder/master?filepath=demo.ipynb)

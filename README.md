[![Build Status](https://dev.azure.com/chembl/FPSim2/_apis/build/status/chembl.FPSim2?branchName=master)](https://dev.azure.com/chembl/FPSim2/_build/latest?definitionId=1&branchName=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Anaconda-Server Badge](https://anaconda.org/efelix/fpsim2/badges/platforms.svg)](https://anaconda.org/efelix/fpsim2)
[![Binder](http://mybinder.org/badge.svg)](http://beta.mybinder.org/v2/gh/eloyfelix/fpsim2_binder/master?filepath=demo.ipynb)
[![DOI](https://zenodo.org/badge/154705090.svg)](https://zenodo.org/badge/latestdoi/154705090)


# FPSim2: Simple package for fast molecular similarity searches

FPSim2 is a small Python/C++ package to run fast compound similarity searches. FPSim2 works better using high search thresholds (>=0.7).

Implementing: 

- A fast population count algorithm using [libpopcnt](https://github.com/kimwalisch/libpopcnt)
- Bounds for sublinear speedups from https://pubs.acs.org/doi/abs/10.1021/ci600358f
- A compressed file format with optimised read speed based in [PyTables](https://www.pytables.org/) and [BLOSC](http://www.blosc.org/pages/blosc-in-depth/)
- In memory and on disk search modes


## Installation (it requires RDKit to work)

From source:

 - clone the repo
 - `pip install ./FPSim2`

From a conda environment:

```bash
conda install -c efelix fpsim2
```

## Usage

### Create FP file

```python
from FPSim2.io import create_db_file

# from .smi file
create_db_file('chembl.smi', 'chembl.h5', 'Morgan', {'radius': 2, 'nBits': 2048})

# from .sdf file, need to specify sdf property containing the molecule id
create_db_file('chembl.sdf', 'chembl.h5', 'Morgan', {'radius': 2, 'nBits': 2048}, mol_id_prop='mol_id')

# from Python list
create_db_file([['CC', 1], ['CCC', 2], ['CCCC', 3]], 'test/10mols.h5', 'Morgan', {'radius': 2, 'nBits': 2048})

# or any other iterable
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

fp_filename = 'chembl_27.h5'
query = 'CC(=O)Oc1ccccc1C(=O)O'

fpe = FPSim2Engine(fp_filename)

results = fpe.similarity(query, 0.7, n_workers=1)
```

As GIL is most of the time released, searches can be speeded up using multiple threads. This is specially useful when dealing with huge datasets and demanding real time results. Performance will vary depending on the population count distribution of the dataset, the query molecule, the threshold, the number of results and the number of threads used.

### Calc the full distance matrix of the database (only in memory mode)

It outputs a SciPy [CSR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) sparse matrix. chembl_27 (1941405 compounds) matrix can be computed in 12.5h with 1 worker using a 0.7 threshold. 3.5h with 4 workers.

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_27.h5'
fpe = FPSim2Engine(fp_filename)

csr_matrix = fpe.symmetric_distance_matrix(0.7, n_workers=1)
```

The CSR distance matrix can be used as an input for some [scikit-learn](https://scikit-learn.org/) algorithms supporting "precomputed" distance metrics.

Some others may need a similarity matrix. A CSR distance matrix can be easily converted to a similarity matrix:

```python
csr_matrix.data = 1 - csr_matrix.data
# 0's in the diagonal of the matrix are implicit so they are not affected by the instruction above
csr_matrix.setdiag(1)
```

Finally, some algorithms (e.g. [MDS](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)) require a dense matrix. Conversion to a dense matrix can be easily done but bear in mind that the number of elements in the dense matrix will be the square of the number of your compounds and this may not fit in your memory.

```python
from sklearn.manifold import MDS

dense_matrix = csr_matrix.todense()

mds = MDS(dissimilarity="precomputed")
pos = mds.fit_transform(dense_matrix)
```

### Run a on disk search

If you're searching against a huge dataset or you have small RAM, you can still run searches.

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_27.h5'
query = 'CC(=O)Oc1ccccc1C(=O)O'

fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

results = fpe.on_disk_similarity(query, 0.7, chunk_size=250000, n_workers=1)
```

In the on disk search variant, parallelisation is achieved with processes. Performance will vary depending on the population count distribution of the dataset, the query molecule, the threshold, the number of results, the chunk size and the number of processes used.

## Trying it online

To try out FPSim2 interactively in your web browser, just click on the binder [![Binder](http://mybinder.org/badge.svg)](http://beta.mybinder.org/v2/gh/eloyfelix/fpsim2_binder/master?filepath=demo.ipynb)

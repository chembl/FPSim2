[![Build Status](https://dev.azure.com/chembl/FPSim2/_apis/build/status/chembl.FPSim2?branchName=master)](https://dev.azure.com/chembl/FPSim2/_build/latest?definitionId=1&branchName=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/fpsim2/badges/platforms.svg)](https://anaconda.org/conda-forge/fpsim2)
[![Binder](http://mybinder.org/badge.svg)](http://beta.mybinder.org/v2/gh/eloyfelix/fpsim2_binder/master?filepath=demo.ipynb)
[![DOI](https://zenodo.org/badge/154705090.svg)](https://zenodo.org/badge/latestdoi/154705090)


# FPSim2: Simple package for fast molecular similarity searches

FPSim2 is a small NumPy centric Python/C++ RDKit based package to run fast compound similarity searches. FPSim2 performs better with high search thresholds (>=0.7). Currently used in the [ChEMBL](http://www.ebi.ac.uk/chembl/) interface.

Highlights:
- Using CPU POPCNT instruction
- Bounds for sublinear speedups from [10.1021/ci600358f](https://pubs.acs.org/doi/abs/10.1021/ci600358f)
- A compressed file format with optimised read speed based in [PyTables](https://www.pytables.org/) and [BLOSC](http://www.blosc.org/pages/blosc-in-depth/)
- Fast multicore CPU and GPU similarity searches
- In memory and on disk search modes
- Distance matrix calculation

## Installation

```bash
conda install -c conda-forge fpsim2
```

## Documentation

Documentation is available at https://chembl.github.io/FPSim2/


## Trying it online

To try out FPSim2 interactively in your web browser, just click on the binder icon [![Binder](http://mybinder.org/badge.svg)](http://beta.mybinder.org/v2/gh/eloyfelix/fpsim2_binder/master?filepath=demo.ipynb)

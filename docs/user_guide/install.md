# Installation

## [Pypi](https://pypi.org/project/FPSim2/)

Builds available for Linux (x86_64, arm64), MacOS (x86_64, arm64) and Windows (x86_64):

```bash
pip install fpsim2
```

## From source

For installation from source, Pybind11 is required as a build dependency.

To potentially enhance CPU similarity searches performance on modern CPUs through compiler optimizations, set the environment variable **FPSIM2_MARCH_NATIVE** to 1 (note that this may not always improve performance depending on your system):

```bash
export FPSIM2_MARCH_NATIVE=1
pip install git+https://github.com/chembl/FPSim2.git
```

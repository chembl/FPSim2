# Installation

## Pypi

Builds available for Linux (x86_64, arm64), MacOS (x86_64, arm64) and Windows (x86_64):

```bash
pip install fpsim2
```

## From source

It can be alternatively installed from source, Pybind11 is required for its build.

clang 10 and above can apply Auto-vectorization on AVX2-capable CPUs, improving performance for Tanimoto searches. To get this performance boost, set the **FPSIM2_MARCH_NATIVE** environment variable to 1:

```bash
export FPSIM2_MARCH_NATIVE=1
pip install git+https://github.com/chembl/FPSim2.git
```

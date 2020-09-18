.. _install:

Installation
============

From conda
----------

Conda builds avaiable for Linux, MacOs and Windws::

    conda install -c conda-forge fpsim2

From source
-----------

It can be alternatively installed from source, Pybind11 is required for its build::

    git clone https://github.com/chembl/FPSim2.git
    pip install FPSim2/

clang 10 and above is able to apply some Auto-vectorization in AVX2 capable CPUs improving by ~25% the performance on Tanimoto searches. To get the boost set **FPSIM2_MARCH_NATIVE** env variable to 1::

    git clone https://github.com/chembl/FPSim2.git
    export FPSIM2_MARCH_NATIVE=1
    pip install FPSim2/

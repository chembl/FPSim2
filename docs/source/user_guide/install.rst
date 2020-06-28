.. _install:

Installation
============

RDKit is used to generate the fingerprints, if you install FPSim2 with conda it will automatically install it. Be sure that you have it installed in case you install from source.


With conda
----------

Conda builds avaiable for Linux, MacOs and Windws::

    conda install -c efelix fpsim2

From source
-----------

It can be also alternatively installed from source, Pybind11 is required for it's build::

    git clone https://github.com/chembl/FPSim2.git
    cd FPSim2
    python setup.py install

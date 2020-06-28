.. _subs:

Run Tversky substructure screenouts
===================================

In memory
---------

Use the :func:`~FPSim2.FPSim2.FPSim2Engine.substructure` function to run a optimised Tversky (a=1, b=0, threshold=1.0) substructure screenout. Bear in mind this is not a full substructure (i.e. with subgraph isomorphism). It's recommended to use **RDKPatternFingerprint** type to run this kind of searches.

.. code-block:: python

    from FPSim2 import FPSim2Engine
    
    fp_filename = 'chembl_27.h5'
    fpe = FPSim2Engine(fp_filename)
    
    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.substructure(query, 0.7, n_workers=1)

On disk
-------

It is also possible to run on disk substructure screenouts (i.e. without loading the whole fingerprints file in memory) with the :func:`~FPSim2.FPSim2.FPSim2Engine.on_disk_substructure` function. This allows running screenouts on databases bigger than the available system memory:

.. code-block:: python

    from FPSim2 import FPSim2Engine

    fp_filename = 'chembl_27.h5'
    fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.on_disk_substructure(query, 0.7, chunk_size=250000, n_workers=1)
.. _sim:

Run Tanimoto similarity searches
================================

In memory
---------

Use the :func:`~FPSim2.FPSim2.FPSim2Engine.similarity` function in the :class:`~FPSim2.FPSim2.FPSim2Engine` class to run a Tanimoto similarity search:

.. code-block:: python

    from FPSim2 import FPSim2Engine
    
    fp_filename = 'chembl_27.h5'
    fpe = FPSim2Engine(fp_filename)
    
    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.similarity(query, 0.7, n_workers=1)


On disk
-------

It is also possible to run on disk similarity searches (i.e. without loading the whole fingerprints file in memory) with the :func:`~FPSim2.FPSim2.FPSim2Engine.on_disk_similarity` function. This allows running similarity searches on databases bigger than the available system memory:

.. code-block:: python

    from FPSim2 import FPSim2Engine

    fp_filename = 'chembl_27.h5'
    fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.on_disk_similarity(query, 0.7, chunk_size=250000, n_workers=1)
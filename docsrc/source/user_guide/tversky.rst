.. _tversky:

Run Tversky searches
====================

In memory
---------

Use the :func:`~FPSim2.FPSim2.FPSim2Engine.tversky` function to run Tversky searches. Tversky is a generalisation of Tanimoto and the Sørensen–Dice coefficients. 

.. tip::
    By setting the a, b and threshold parameters:
        - a=b=1 will return the same results than a Tanimoto search but will be slower than using :func:`~FPSim2.FPSim2.FPSim2Engine.similarity`.
        - a=b=0.5: will run a Sørensen–Dice search.
        - a=1, b=0, threshold=1.0 will return the same results than using :func:`~FPSim2.FPSim2.FPSim2Engine.substructure` function but will be slower.

.. code-block:: python

    from FPSim2 import FPSim2Engine
    
    fp_filename = 'chembl_27.h5'
    fpe = FPSim2Engine(fp_filename)
    
    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.tversky(query, 0.7, 0.5, 0.5, n_workers=1)

.. tip::
    *n_workers* parameter can be used to split a single query into multiple threads to speed up the seach. This is specially useful on big datasets.

On disk
-------

It is also possible to run on disk similarity searches (i.e. without loading the whole fingerprints file in memory) with the :func:`~FPSim2.FPSim2.FPSim2Engine.on_disk_tversky` function. This allows running similarity searches on databases bigger than the available system memory:

.. code-block:: python

    from FPSim2 import FPSim2Engine

    fp_filename = 'chembl_27.h5'
    fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.on_disk_tversky(query, 0.7, 0.5, 0.5, n_workers=1)
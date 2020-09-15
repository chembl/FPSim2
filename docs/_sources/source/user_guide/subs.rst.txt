.. _subs:

Run Tversky substructure screenouts
===================================

In memory
---------

Use the :func:`~FPSim2.FPSim2.FPSim2Engine.substructure` function to run an optimised Tversky (*a=1*, *b=0*, *threshold=1.0*) substructure screenout. Bear in mind that this is not a full substructure (i.e. with subgraph isomorphism) search.

.. tip::
    It's recommended to use **RDKPatternFingerprint** fingerprint type with this kind of searches.

.. code-block:: python

    from FPSim2 import FPSim2Engine
    
    fp_filename = 'chembl_27.h5'
    fpe = FPSim2Engine(fp_filename)
    
    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.substructure(query, n_workers=1)

.. tip::
    *n_workers* parameter can be used to split a single query into multiple threads to speed up the seach. This is specially useful on big datasets.

On disk
-------

It is also possible to run on disk substructure screenouts (i.e. without loading the whole fingerprints file in memory) with the :func:`~FPSim2.FPSim2.FPSim2Engine.on_disk_substructure` function. This allows running screenouts on databases bigger than the available system memory:

.. code-block:: python

    from FPSim2 import FPSim2Engine

    fp_filename = 'chembl_27.h5'
    fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.on_disk_substructure(query, n_workers=1)
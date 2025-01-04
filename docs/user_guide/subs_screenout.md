# Substructure Screenouts

Run an optimised Tversky (a=1, b=0, threshold=1.0) substructure screenout. Note that this is not a full substructure (i.e., with subgraph isomorphism) search.

!!! tip
    It's recommended to use **RDKitPattern** fingerprint type with this kind of searches.

=== "In memory"
    Use the `FPSim2Engine.substructure` function to run a substructure screenout.
    ```python
    from FPSim2 import FPSim2Engine

    fp_filename = 'chembl_35_v0.6.0.h5'
    fpe = FPSim2Engine(fp_filename)

    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.substructure(query, n_workers=1)
    ```

=== "On disk"
    Use the `FPSim2Engine.on_disk_substructure` function to run an on disk substructure screenout (much slower but doesn't require loading the entire fingerprint file into memory).

    ```python
    from FPSim2 import FPSim2Engine

    fp_filename = 'chembl_35_v0.6.0.h5'
    fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.on_disk_substructure(query, n_workers=1)
    ```





!!! tip "Parallel Processing"
    The `n_workers` parameter can be used to split a single query into multiple threads to speed up the search. This is especially useful when searching large datasets.

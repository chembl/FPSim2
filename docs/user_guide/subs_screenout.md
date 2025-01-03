# Run Tversky Substructure Screenouts

## In memory

Use the `FPSim2Engine.substructure` function to run an optimised Tversky (*a=1*, *b=0*, *threshold=1.0*) substructure screenout. Bear in mind that this is not a full substructure (i.e. with subgraph isomorphism) search.

!!! tip
    It's recommended to use **RDKitPattern** fingerprint type with this kind of searches.

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename)

query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.substructure(query, n_workers=1)
```

!!! tip "Parallel Processing"
    The `n_workers` parameter can be used to split a single query into multiple threads to speed up the search. This is especially useful when searching large datasets.

## On disk

For on-disk substructure screenouts (slower but doesn't require loading the entire fingerprint file into memory), use the `FPSim2Engine.on_disk_substructure` function. This is useful for databases larger than the available system memory:

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.on_disk_substructure(query, n_workers=1)
```

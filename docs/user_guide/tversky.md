# Run Tversky searches

## In memory

Use the `FPSim2Engine.tversky` function to run Tversky searches. Tversky is a generalisation of Tanimoto and the Sørensen–Dice coefficients.

!!! tip "Search Types Configuration"
    Set `a=b=0.5` to perform a Sørensen–Dice search.

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename)

query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.tversky(query, threshold=0.7, a=0.5, b=0.5, n_workers=1)
```

!!! tip "Parallel Processing"
    The `n_workers` parameter can be used to split a single query into multiple threads to speed up the search.
    This is especially useful when searching large datasets.

## On disk

It is also possible to run on disk similarity searches (i.e. without loading the whole fingerprints file in memory) with the `FPSim2Engine.on_disk_tversky` function. This allows running similarity searches on databases bigger than the available system memory:

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.on_disk_tversky(query, threshold=0.7, a=0.5, b=0.5, n_workers=1)
```

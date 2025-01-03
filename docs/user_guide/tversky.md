# FPSim2 Tversky Searches

## In memory

Use the `FPSim2Engine.tversky` function to run Tversky asymmetric searches. The Tversky similarity coefficient is a generalization of the Tanimoto coefficient that allows asymmetric weighting between query and reference fingerprints.


!!! info "Using `a` and `b` Parameters"
    The `a` and `b` parameters in the `tversky` function control the weighting of the query and reference fingerprints, respectively. Adjusting these values allows you to fine-tune the similarity measure to emphasize different aspects of the fingerprints. For example, setting `a` to a higher value than `b` will give more weight to the query fingerprint.

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename)

query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.tversky(query, threshold=0.7, a=0.7, b=0.3, n_workers=1)
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
results = fpe.on_disk_tversky(query, threshold=0.7, a=0.7, b=0.3, n_workers=1)
```

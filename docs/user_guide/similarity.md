# Run Tanimoto similarity searches

## In memory

Use the `similarity` function in the `FPSim2Engine` class to run a Tanimoto similarity search:

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename)

query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.similarity(query, threshold=0.7, n_workers=1)
```


To search for the top K most similar molecules use the `top_k` function:

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename)

query = 'CC(=O)Oc1ccccc1C(=O)O'
# Find top 100 most similar molecules with min threshold 0.7
results = fpe.top_k(query, k=100, threshold=0.7, n_workers=1)
```

!!! tip
    `n_workers` parameter can be used to split a single query into multiple threads to speed up the search. This is specially useful on big datasets.

## On disk

It is also possible to run on disk similarity searches (i.e. without loading the whole fingerprints file in memory) with the `on_disk_similarity` function. This allows running similarity searches on databases bigger than the available system memory:

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.on_disk_similarity(query, threshold=0.7, n_workers=1)
```

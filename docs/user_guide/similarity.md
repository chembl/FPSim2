# FPSim2 Symmetric Similarity Searches

## In memory

Use the `FPSim2Engine.similarity` function to run symmetric similarity searches.

!!! info "Similarity Metrics"
    Possible metrics that can be used are (`tanimoto` is default):

    - `tanimoto` (Jaccard): Measures the ratio of intersection to union. Commonly used for binary fingerprints, providing a balance between shared and distinct features.
    - `dice` (Dice-Sørensen): Emphasizes the intersection more than Tanimoto. Useful when you want to highlight common features between fingerprints.
    - `cosine` (Otsuka–Ochiai): Also focuses on shared features but is less affected by the total number of features, making it more robust when comparing sparse fingerprints.

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename)
query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.similarity(query, threshold=0.7, metric='tanimoto', n_workers=1)
```

!!! tip "Parallel Processing"
    The `n_workers` parameter can be used to split a single query into multiple threads to speed up the search. This is especially useful when searching large datasets.

## On disk

For on-disk similarity searches (slower but doesn't require loading the entire fingerprint file into memory), use the `FPSim2Engine.on_disk_similarity` function. This is useful for databases larger than the available system memory:

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.on_disk_similarity(query, threshold=0.7, metric='tanimoto', n_workers=1)
```

## Top K Searches

Retrieve the top K most similar hits using the `FPSim2Engine.top_k` function. You can specify a different similarity metric:

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename)

query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.top_k(query, k=100, threshold=0.7, metric='tanimoto', n_workers=1)
```

## On disk

For on-disk top K searches, use the `FPSim2Engine.on_disk_top_k` function:

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.on_disk_top_k(query, k=100, threshold=0.7, metric='tanimoto', n_workers=1)
```

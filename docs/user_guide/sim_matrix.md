# Computing a Symmetric Distance Matrix


Use the `FPSim2.FPSim2Engine.symmetric_distance_matrix` function to create a SciPy CSR sparse distance matrix from the current dataset:

!!! info "Similarity Metrics"
    Possible metrics that can be used are (`tanimoto` is default):

    - `tanimoto` (Jaccard): Measures the ratio of intersection to union. Commonly used for binary fingerprints, providing a balance between shared and distinct features.
    - `dice` (Dice-Sørensen): Emphasizes the intersection more than Tanimoto. Useful when you want to highlight common features between fingerprints.
    - `cosine` (Otsuka–Ochiai): Also focuses on shared features but is less affected by the total number of features, making it more robust when comparing sparse fingerprints.

```python
from FPSim2 import FPSim2Engine

fp_filename = 'chembl_35_v0.6.0.h5'
fpe = FPSim2Engine(fp_filename)
csr_matrix = fpe.symmetric_distance_matrix(threshold=0.7, metric="tanimoto", n_workers=4)
```

!!! note
    Code above generating the symmetric matrix of ChEMBL 27 (1941405 compounds) took 3.5h, using 4 cores, running in a 2019 core i9 laptop. 12.5h in a single core.

The order of the compounds is the same one than in the fps file (the compounds get sorted by number of fingerprint features). To get the fps ids:

```python
ids = fpe.fps[:, 0]
```

The CSR distance matrix can be used as an input for some scikit-learn algorithms supporting "precomputed" distance metrics. Some others may need a similarity matrix. A CSR distance matrix can be easily converted into a similarity matrix:

```python
csr_matrix.data = 1 - csr_matrix.data
# 0's in the diagonal of the matrix are implicit so they are not affected by the instruction above
csr_matrix.setdiag(1)
```

Finally, some algorithms (e.g. MDS) require a dense matrix. Conversion to a dense matrix can be easily done but bear in mind that the number of elements in the dense matrix will be the square of the number of your compounds and this may not fit in your memory:

```python
# classic MDS doesn't work with missing values, so it's better to only use it with threshold 0.0
# in case you still want to run MDS on missing values matrices
# this example uses the SMACOF algorithm which is known for being able to deal with missing data. 
# Use it at your own risk!

from sklearn.manifold import MDS

dense_matrix = csr_matrix.todense()

# with metric=False it uses the SMACOF algorithm
mds = MDS(dissimilarity="precomputed", metric=False)
pos = mds.fit_transform(dense_matrix)
```

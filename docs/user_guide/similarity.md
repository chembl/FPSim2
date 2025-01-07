# Symmetric Similarity Searches

Run a similarity search to find compounds that are structurally similar to a query molecule.

!!! info "Similarity Metrics"
    Possible metrics that can be used are (`tanimoto` is default):

    - `tanimoto` (Jaccard): Measures the ratio of intersection to union. $T(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{c}{a + b - c}$
    - `dice` (Dice-Sørensen): Emphasizes the intersection more than Tanimoto. $D(A,B) = \frac{2|A \cap B|}{|A| + |B|} = \frac{2c}{a + b}$
    - `cosine` (Otsuka–Ochiai): Also focuses on shared features but is less affected by the total number of features. $C(A,B) = \frac{|A \cap B|}{\sqrt{|A| \cdot |B|}} = \frac{c}{\sqrt{a \cdot b}}$

    Where:

    - $a$ is the number of bits set to 1 in fingerprint A
    - $b$ is the number of bits set to 1 in fingerprint B
    - $c$ is the number of bits set to 1 in both fingerprints


=== "In memory"
    Use the `FPSim2Engine.similarity` function to run symmetric similarity searches:
    ```python
    from FPSim2 import FPSim2Engine

    fp_filename = 'chembl_35_v0.6.0.h5'
    fpe = FPSim2Engine(fp_filename)

    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.similarity(query, threshold=0.7, metric='tanimoto', n_workers=1)
    ```

=== "On disk"
    For on-disk similarity searches (slower but doesn't require loading the entire fingerprint file into memory), use the `FPSim2Engine.on_disk_similarity` function. This is useful for databases larger than the available system memory:

    ```python
    from FPSim2 import FPSim2Engine

    fp_filename = 'chembl_35_v0.6.0.h5'
    fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.on_disk_similarity(query, threshold=0.7, metric='tanimoto', n_workers=1)
    ```

!!! tip "Parallel Processing"
    The `n_workers` parameter can be used to split a single query into multiple threads to speed up the search. This is especially useful when searching large datasets.


## Top K Searches

=== "In memory"
    Retrieve the top K most similar hits using the `FPSim2Engine.top_k` function.
    ```python
    from FPSim2 import FPSim2Engine

    fp_filename = 'chembl_35_v0.6.0.h5'
    fpe = FPSim2Engine(fp_filename)

    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.top_k(query, k=100, threshold=0.7, metric='tanimoto', n_workers=1)
    ```

=== "On disk"
    For on-disk top K searches, use the `FPSim2Engine.on_disk_top_k` function.

    ```python
    from FPSim2 import FPSim2Engine

    fp_filename = 'chembl_35_v0.6.0.h5'
    fpe = FPSim2Engine(fp_filename, in_memory_fps=False)

    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpe.on_disk_top_k(query, k=100, threshold=0.7, metric='tanimoto', n_workers=1)
    ```

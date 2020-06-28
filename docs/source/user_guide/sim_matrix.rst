.. _sim_mat:

Compute a symmetric similarity matrix
=====================================

Use the :func:`~FPSim2.FPSim2.FPSim2Engine.symmetric_distance_matrix` function to create a SciPy CSR sparse distance matrix from the current dataset::

    >>> from FPSim2 import FPSim2CudaEngine
    >>> fp_filename = 'chembl_27.h5'
    >>> fpe = FPSim2CudaEngine(fp_filename)
    >>> csr_matrix = fpe.symmetric_distance_matrix(0.7)


The CSR distance matrix can be used as an input for some scikit-learn algorithms supporting "precomputed" distance metrics. Some others may need a similarity matrix. A CSR distance matrix can be easily converted into a similarity matrix:


.. code-block:: python

    csr_matrix.data = 1 - csr_matrix.data
    # 0's in the diagonal of the matrix are implicit so they are not affected by the instruction above
    csr_matrix.setdiag(1)

Finally, some algorithms (e.g. MDS) require a dense matrix. Conversion to a dense matrix can be easily done but bear in mind that the number of elements in the dense matrix will be the square of the number of your compounds and this may not fit in your memory:

.. code-block:: python

    # classic MDS doesn't work with missing values, so it's better to only use it with threshold 0.0
    # in case you still want to run MDS on missing values matrices
    # this example uses the SMACOF algorithm which is known for being able to deal with missing data. 
    # Use it at your own risk!

    from sklearn.manifold import MDS

    dense_matrix = csr_matrix.todense()

    # with metric=False it uses the SMACOF algorithm
    mds = MDS(dissimilarity="precomputed", metric=False)
    pos = mds.fit_transform(dense_matrix)

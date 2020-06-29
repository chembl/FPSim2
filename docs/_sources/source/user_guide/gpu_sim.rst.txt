.. _gpu_sim:

Run GPU similarity searches
===========================

Experimental CUDA similarity search is implemented via the `CuPy <https://cupy.chainer.org/>`_ library.

.. important::
    Be sure to have CuPy properly `installed <https://docs-cupy.chainer.org/en/stable/install.html>`_  before using this feature.

Use the :func:`~FPSim2.FPSim2Cuda.FPSim2CudaEngine.similarity` function in the :class:`~FPSim2.FPSim2Cuda.FPSim2CudaEngine` class to run a Tanimoto similarity search:

.. code-block:: python

    from FPSim2 import FPSim2CudaEngine

    fp_filename = 'chembl_27.h5'
    fpce = FPSim2CudaEngine(fp_filename)

    query = 'CC(=O)Oc1ccccc1C(=O)O'
    results = fpce.similarity(query, 0.7)

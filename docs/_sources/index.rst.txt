.. FPSim2 documentation master file, created by
   sphinx-quickstart on Sat Jun 27 23:03:18 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FPSim2's documentation!
==================================

FPSim2 is a small NumPy centric Python/C++ package to run fast compound similarity searches. Currently used in the `ChEMBL
<http://www.ebi.ac.uk/chembl/>`_ interface. 

Highlights
----------

 - Runs fast in memory CPU similarity searches. 
 - Can split a single query in multiple CPU threads to accelerate a search. 
 - Can run similarities in a GPU using CUDA via the CuPy library, using the :class:`~FPSim2.FPSim2Cuda.FPSim2CudaEngine`. 
 - Can be used to run on disk CPU searches in case there is not enough RAM to load the dataset.


Getting started
---------------

Install and generate a FPSim2 fingerprint db file.

* **Getting started**:
  :doc:`Installation </source/user_guide/install>` |
  :doc:`Create a fingeprint db file </source/user_guide/create_fp_db>` |
  :doc:`Limitations </source/user_guide/create_fp_db>`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting started

   source/user_guide/install
   source/user_guide/create_fp_db
   source/user_guide/limitations


CPU searches
------------

Learn how to run CPU based similarity searches


* **CPU searches**:
  :doc:`Run Tanimoto similarity searches </source/user_guide/sim>` |
  :doc:`Run a Tversky search with a and b parameters </source/user_guide/tversky>` |
  :doc:`Run Tversky substructure screenouts </source/user_guide/subs>` |
  :doc:`Generate a symmetics similarity matrix </source/user_guide/sim_matrix>`


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: CPU searches

   source/user_guide/sim
   source/user_guide/tversky
   source/user_guide/subs
   source/user_guide/sim_matrix


GPU searches
------------

Learn how to run GPU based similarity searches

* **GPU searches**:
  :doc:`Run GPU based Tanimoto similarity searches </source/user_guide/gpu_sim>`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: GPU searches

   source/user_guide/gpu_sim


API documentation
==================

* :ref:`genindex`
* :ref:`modindex`

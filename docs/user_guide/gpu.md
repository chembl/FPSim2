# Run GPU similarity searches

Experimental CUDA similarity search is implemented via the [CuPy](https://cupy.chainer.org/) library.

!!! important
    Be sure to have CuPy properly [installed](https://docs-cupy.chainer.org/en/stable/install.html) before using this feature.

Use the `similarity` function in the `FPSim2CudaEngine` class to run a Tanimoto similarity search:

```python
from FPSim2 import FPSim2CudaEngine

fp_filename = 'chembl_27.h5'
fpce = FPSim2CudaEngine(fp_filename)

query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpce.similarity(query, threshold=0.7)
```

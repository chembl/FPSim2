# Conda recipe for FPSim2 package

To build the packages:

```
conda build --py 3.6 conda_recipe/ -c rdkit
```

To upload to conda cloud:
```
anaconda upload ./.../fpsim2-0.16-py37_0.tar.bz2
```
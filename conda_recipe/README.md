# Conda recipe for FPSim2 package

To build the packages:

Linux:
```
conda build --py 3.5 --py 3.6 --py 3.7 fpsim2/ -c conda-forge
```

Mac (there are some problems with python 3.7 and RDKit):
```
conda build --py 3.5 --py 3.6 fpsim2/ -c conda-forge
```

To upload to conda cloud:
```
anaconda upload ./.../fpsim2-0.16-py37_0.tar.bz2
```
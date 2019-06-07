# Conda recipe for FPSim2 package

To build the packages:

```
conda build --py 3.6 --py 3.7 conda.recipe/ 
```

To upload to conda cloud:
```
anaconda upload ./.../fpsim2-0.1.0*.tar.bz2
```
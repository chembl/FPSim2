from .FPSim2 import FPSim2Engine

try:
    from .FPSim2Cuda import FPSim2CudaEngine
except Exception as e:
    pass

__version__ = "0.2.9"
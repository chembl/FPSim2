from .FPSim2 import FPSim2Engine

try:
    from .FPSim2Cuda import FPSim2CudaEngine
except Exception:
    pass

try:
    from .FPSim2Metal import FPSim2MetalEngine
except Exception:
    pass

try:
    from importlib.metadata import version
    __version__ = version("FPSim2")
except ImportError:
    __version__ = "unknown"
from abc import ABC, abstractmethod
from .io.chem import load_molecule, build_fp
from .io.backends import PyTablesStorageBackend
from .FPSim2lib.utils import PyPopcount
import numpy as np


class BaseEngine(ABC):

    fp_filename = None
    storage = None

    def __init__(
        self,
        fp_filename: str,
        storage_backend: str,
        in_memory_fps: bool,
        fps_sort: bool,
    ) -> None:

        self.fp_filename = fp_filename
        self.in_memory_fps = in_memory_fps
        if storage_backend == "pytables":
            self.storage = PyTablesStorageBackend(
                fp_filename, in_memory_fps=in_memory_fps, fps_sort=fps_sort
            )

    @property
    def fps(self):
        if self.in_memory_fps:
            return self.storage.fps
        else:
            raise Exception("FPs not loaded into memory.")

    @property
    def popcnt_bins(self):
        return self.storage.popcnt_bins

    @property
    def fp_type(self):
        return self.storage.fp_type

    @property
    def fp_params(self):
        return self.storage.fp_params

    @property
    def rdkit_ver(self):
        return self.storage.rdkit_ver

    def load_query(self, query_string: str) -> np.ndarray:
        """Loads the query molecule from SMILES, molblock or InChI.

        Parameters
        ----------
        query_string : str
            SMILES, InChi or molblock.

        Returns
        -------
        query : numpy array
            Numpy array query molecule.
        """
        rdmol = load_molecule(query_string)
        fp = build_fp(rdmol, self.fp_type, self.fp_params, 0)
        return np.array(fp, dtype=np.uint64)

    @abstractmethod
    def similarity(
        self, query_string: str, threshold: float, n_workers=1
    ) -> np.ndarray:
        """Tanimoto similarity search """

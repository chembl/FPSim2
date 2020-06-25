from abc import ABC, abstractmethod
from .io.chem import load_molecule, rdmol_to_efp
from .io.backends import PyTablesStorageBackend
from .FPSim2lib import py_popcount
import numpy as np

SEARCH_TYPES = {"tanimoto": 0, "tversky": 1, "substructure": 2}


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
        if storage_backend == "pytables":
            self.storage = PyTablesStorageBackend(
                fp_filename, in_memory_fps=in_memory_fps, fps_sort=fps_sort
            )

    @property
    def fps(self):
        return self.storage.fps

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
        """Load query molecule from SMILES, molblock or InChi.

        Args:
            query_string: SMILES, molblock or InChi.
        Returns:
            Numpy array query molecule.
        """
        rdmol = load_molecule(query_string)
        # generate the efp
        efp = rdmol_to_efp(rdmol, self.fp_type, self.fp_params)
        efp.append(py_popcount(np.array(efp, dtype=np.uint64)))
        efp.insert(0, 0)
        return np.array(efp, dtype=np.uint64)

    @abstractmethod
    def similarity(
        self, query_string: str, threshold: float, n_workers=1
    ) -> np.ndarray:
        """ Tanimoto similarity search """

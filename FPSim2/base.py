from abc import ABC, abstractmethod
from .io.chem import load_molecule, build_fp
from .io.backends import PyTablesStorageBackend
from .io.backends import SqlaStorageBackend
from sqlalchemy import create_mock_engine
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
        conn_url: str,
        table_name: str,
    ) -> None:

        self.fp_filename = fp_filename
        self.in_memory_fps = in_memory_fps
        if storage_backend == "pytables":
            if not fp_filename:
                raise ValueError(
                    "Missing required 'fp_filename' param for the pytables backend"
                )
            self.storage = PyTablesStorageBackend(
                fp_filename, in_memory_fps=in_memory_fps, fps_sort=fps_sort
            )
        elif storage_backend == "sqla":
            if not conn_url or not table_name:
                raise ValueError(
                    "Missing required 'conn_url' or 'table_name' param for the sqla backend"
                )
            engine = create_mock_engine(conn_url, ())
            if engine.dialect.name not in ("postgresql", "mysql"):
                raise ValueError(
                    "FPSim2 sqla engine only works for PostgreSQL and MySQL"
                )
            self.storage = SqlaStorageBackend(conn_url, table_name)

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
        """Tanimoto similarity search"""

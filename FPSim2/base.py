from abc import ABC, abstractmethod
from .io.chem import load_molecule, build_fp, process_fp
from .io.backends.pytables import create_schema, get_fp_length
from .io.backends import PyTablesStorageBackend
from .io.backends import SqlaStorageBackend
from sqlalchemy import create_mock_engine
from rdkit.DataStructs import ExplicitBitVect
from rdkit import Chem
from typing import Union
import tables as tb
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
        pg_schema: str,
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
            if engine.dialect.name not in ("postgresql", "mysql", "oracle"):
                raise ValueError(
                    "FPSim2 sqla engine only works for PostgreSQL, MySQL and Oracle (experimental)"
                )
            self.storage = SqlaStorageBackend(conn_url, table_name, pg_schema)

    def __str__(self):
        return f"FPSim2Engine(fp_type='{self.fp_type}', fp_params={self.fp_params}, rdkit_ver={self.rdkit_ver}, fpsim2_ver={self.fpsim2_ver})"

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

    @property
    def fpsim2_ver(self):
        return self.storage.fpsim2_ver

    def load_query(
        self,
        query: Union[str, ExplicitBitVect, Chem.Mol],
        full_sanitization: bool = True,
    ) -> np.ndarray:
        """Loads the query fingerprint from SMILES, molblock, InChI or ExplicitBitVect fingerprint.

        Parameters
        ----------
        query : Union[str, ExplicitBitVect]
            SMILES, InChi, molblock or fingerprint as ExplicitBitVect.

        Returns
        -------
        query : numpy array
            Numpy array query molecule.
        """

        if isinstance(query, ExplicitBitVect):
            fp = process_fp(query, 0)
        elif isinstance(query, Chem.Mol):
            fp = build_fp(query, self.fp_type, self.fp_params, 0)
        else:
            rdmol = load_molecule(query, full_sanitization=full_sanitization)
            fp = build_fp(rdmol, self.fp_type, self.fp_params, 0)
        return np.array(fp, dtype=np.uint64)

    @abstractmethod
    def similarity(
        self, query_string: str, threshold: float, n_workers=1
    ) -> np.ndarray:
        """Tanimoto similarity search"""

    def save_h5(self, filename: str) -> None:
        """Save the fingerprints to a HDF5 file.

        Useful when the fingerprints were loaded from SQL and want to save them to a file.
        """

        if not self.in_memory_fps:
            raise Exception("FPs not loaded into memory.")

        if not hasattr(self, "popcnt_bins"):
            raise Exception(
                "FPs are not sorted and popcnt_bins are not stored in memory."
            )

        fp_length = get_fp_length(self.fp_type, self.fp_params)
        filters = tb.Filters(complib="blosc2", complevel=9, fletcher32=False)

        with tb.open_file(filename, mode="w") as out_file:
            table_class = create_schema(fp_length)
            fps_table = out_file.create_table(
                out_file.root, "fps", table_class, "Table storing fps", filters=filters
            )
            config_table = out_file.create_vlarray(
                out_file.root, "config", atom=tb.ObjectAtom()
            )
            config_table.append(self.fp_type)
            config_table.append(self.fp_params)
            config_table.append(self.rdkit_ver)
            config_table.append(self.fpsim2_ver)

            if self.fps is not None:
                fps_table.append(self.fps)
                config_table.append(self.popcnt_bins)

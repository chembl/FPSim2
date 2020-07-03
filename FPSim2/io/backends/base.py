import numpy as np
from abc import ABC, abstractmethod


class BaseStorageBackend(ABC):

    def __init__(self, fp_filename):
        self.fp_filename = fp_filename

    @staticmethod
    def calc_popcnt_bins(fps: np.ndarray) -> list:
        """Calcs popcount bins.

        Parameters
        ----------
        fps : numpy array
            numpy array with the fingerprints of the database.

        Returns
        -------
        popcnt_bins: list
            list with the ranges of the popcount bins.
        """
        popcnt_bins = []
        idx = np.unique(fps[:, -1], return_index=True)
        for i, k in enumerate(zip(*idx)):
            if k[0] == idx[0][-1]:
                popcnt_bins.append((k[0], (k[1], fps.shape[0])))
            else:
                popcnt_bins.append((k[0], (k[1], idx[1][int(i + 1)])))
        return popcnt_bins

    @abstractmethod
    def load_fps(self, in_memory_fps, fps_sort) -> None:
        """Loads the whole fp db file into memory"""

    @abstractmethod
    def read_parameters(self):
        """Reads fp file info"""

    @abstractmethod
    def load_popcnt_bins(self):
        """Loads popcnt bins info"""

    @abstractmethod
    def get_fps_chunk(self, chunk_range):
        """Loads a contiguous chunk of data from the storage into memory"""

    @abstractmethod
    def delete_fps(self, ids_list):
        """Delete FPs given a list of ids.

        Parameters
        ----------
        ids_list : list
            ids to delete.

        Returns
        -------
        None
        """

    @abstractmethod
    def append_fps(self, io_source, mol_id_prop):
        """Appends FPs to the file.

        Parameters
        ----------
        mols_source : str or iterable
            .smi or .sdf filename or iterable.

        Returns
        -------
        None
        """

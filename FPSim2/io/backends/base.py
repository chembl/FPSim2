import numpy as np
from abc import ABC, abstractmethod


class BaseStorageBackend(ABC):

    def __init__(self, fp_filename):
        self.fp_filename = fp_filename

    @staticmethod
    def calc_popcnt_bins(fps: np.ndarray, fp_length: int) -> list:
        """Calcs popcount bins

        Args:
            fps: np array storing fps
            fp_length: length of the fp
            kwargs: keyword arguments
        Returns:
            list with popcnt ranges
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
        """Delete fps from FP db file
        Args:
            ids_list: ids to delete list
        Returns:
            None
        """

    @abstractmethod
    def append_fps(self, io_source, mol_id_prop="mol_id"):
        """Appends new fps to the fp db file
        Args:
            mols_source: .smi or .sdf filename or iterable
            mol_id_prop: name of the property storing the id in sdf files
        Returns:
            None
        """

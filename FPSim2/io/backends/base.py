

class BaseStorageBackend:

    def __init__(self, fp_filename, in_memory_fps=True, fps_sort=False):
        self.fp_filename = fp_filename
        self.in_memory_fps = in_memory_fps
        self.fps_sort = fps_sort

    def read_parameters(self):
        """Reads file info"""
        pass

    def get_count_ranges(self):
        pass

    def get_fps_chunk(self, chunk_range):
        pass

    def load_fps(self):
        """Loads FP db file into memory.
        Uses self.fps_sort to decide wheter it needs to sort them after loading or not.
        """
        pass

    def delete_fps(self, ids_list):
        """Delete fps from FP db file.

        Args:
            ids_list: ids to delete list.
        Returns:
            None.
        """
        pass

    def append_fps(self, io_source, mol_id_prop="mol_id"):
        """Appends fps to a FP db file.

        Args:
            io_source: .smi or .sdf filename, ResultProxy or list.
            mol_id_prop: name of the property storing the id in sdf files.
        Returns:
            None.
        """
        pass

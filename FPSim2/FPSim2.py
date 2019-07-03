import concurrent.futures as cf
import numpy as np
import tables as tb
from .io import S_INDEXS, load_fps, load_query, get_bounds_range
from .FPSim2lib import _similarity_search, _substructure_search, sort_results


def on_disk_search(query, fp_filename, chunk_indexes, threshold, s_index):
    """Run a on disk search.

        Args:
            query: Preprocessed NumPy query array
            fp_filename: Flat to load into memory or not the fps
            chunk_indexes: List with the start and the end of the chunk to search.
            threshold: Search threshold.
            s_index: Type of search.
        Returns:
            Numpy array with results.
    """
    with tb.open_file(fp_filename, mode="r") as fp_file:
        fps = fp_file.root.fps[chunk_indexes[0] : chunk_indexes[1]]
    num_fields = len(fps[0])
    fps = fps.view("<u8")
    fps = fps.reshape(int(fps.size / num_fields), num_fields)
    if s_index == 2:
        res = _substructure_search(query, fps, threshold, 0, fps.shape[0])
    else:
        res = _similarity_search(query, fps, threshold, 0, fps.shape[0])
    return res


class FPSim2Engine:
    """FPSim2 base class to run searches.

    Args:
        fp_filename: FP db file path.
        in_memory_fps: Flat to load into memory or not the fps.
        fps_sort: Flag to sort or not fps after loading into memory.

    Attributes:
        fp_filename: FP db file path.
        fp_type: FP type used to generate the fingerprints.
        fp_params: Parameters used in the fingerprint function.
        rdkit_ver: RDKit version used to generate the fingerprints.
    """

    fp_filename = None
    fp_type = None
    fp_params = None
    rdkit_ver = None
    fps = None

    def __init__(self, fp_filename, in_memory_fps=True, fps_sort=False):
        self.fp_filename = fp_filename
        with tb.open_file(fp_filename, mode="r") as fp_file:
            self.fp_type = fp_file.root.config[0]
            self.fp_params = fp_file.root.config[1]
            self.rdkit_ver = fp_file.root.config[2]
        if in_memory_fps:
            self.load_fps(fps_sort)

    def load_fps(self, fps_sort=False):
        """Loads FP db file into memory.

        Args:
            fps_sort: Sort fps in memory after loading them.
        Returns:
            None.
        """
        self.fps = load_fps(self.fp_filename, fps_sort)

    def similarity(self, query_string, threshold, n_workers=1):
        """Run a similarity search

        Args:
            query_string: SMILES, InChi or molblock.
            threshold: Similarities with a coeff above the threshold will be kept.
            n_workers: Number of threads used for the search.
        Raises:
            Exception: If fps are not loaded into memory.
        Returns:
            Numpy array with ids and similarities.
        """
        if not self.fps:
            raise Exception(
                "Load the fingerprints into memory before running a in memory search"
            )
        return self._base_search(
            query_string,
            threshold,
            _similarity_search,
            0,
            "tanimoto",
            False,
            cf.ThreadPoolExecutor,
            n_workers,
        )

    def on_disk_similarity(
        self, query_string, threshold, n_workers=1, chunk_size=250000
    ):
        """Run a on disk similarity search.

        Args:
            query_string: SMILES, InChi or molblock.
            threshold: Similarities with a coeff above the threshold will be kept.
            n_workers: Number of threads used for the search.
            chunk_size: Chunk size.
        Returns:
            Numpy array with ids and similarities.
        """
        return self._base_search(
            query_string,
            threshold,
            on_disk_search,
            chunk_size,
            "tanimoto",
            True,
            cf.ProcessPoolExecutor,
            n_workers,
        )

    def substructure(self, query_string, n_workers=1):
        """Run a substructure screenout

        Args:
            query_string: SMILES, InChi or molblock.
            n_workers: Number of processes used for the search.
        Raises:
            Exception: If fps are not loaded into memory.
        Returns:
            NumPy array with ids.
        """
        if not self.fps:
            raise Exception(
                "Load the fingerprints into memory before running a in memory search"
            )
        return self._base_search(
            query_string,
            1.0,
            _substructure_search,
            0,
            "substructure",
            False,
            cf.ThreadPoolExecutor,
            n_workers,
        )

    def on_disk_substructure(self, query_string, n_workers=1, chunk_size=250000):
        """Run a on disk substructure screenout.

        Args:
            query_string: SMILES, InChi or molblock.
            n_workers: Number of processes used for the search.
            chunk_size: Chunk size.
        Returns:
            NumPy array with ids.
        """
        return self._base_search(
            query_string,
            1.0,
            on_disk_search,
            chunk_size,
            "substructure",
            True,
            cf.ProcessPoolExecutor,
            n_workers,
        )

    def _load_query_and_fp_range(self, query_string, count_ranges, threshold, s_index):
        query = load_query(query_string, self.fp_filename)
        fp_range = get_bounds_range(query, count_ranges, threshold, S_INDEXS[s_index])
        return query, fp_range

    def _parallel_run(
        self,
        query,
        search_func,
        executor,
        fp_range,
        n_workers,
        threshold,
        chunk_size,
        s_index,
        on_disk,
    ):
        i_start = fp_range[0]
        i_end = fp_range[1]
        results = []
        with executor(max_workers=n_workers) as exe:
            if not on_disk:
                chunk_size = int((i_end - i_start) / n_workers)
            c_indexes = [[x, x + chunk_size] for x in range(i_start, i_end, chunk_size)]
            c_indexes[-1][1] = i_end
            if on_disk:
                future_ss = {
                    exe.submit(
                        search_func,
                        query,
                        self.fp_filename,
                        indexes,
                        threshold,
                        S_INDEXS[s_index],
                    ): c_id
                    for c_id, indexes in enumerate(c_indexes)
                }
            else:
                future_ss = {
                    exe.submit(
                        search_func,
                        query,
                        self.fps.fps,
                        threshold,
                        chunk_idx[0],
                        chunk_idx[1],
                    ): c_id
                    for c_id, chunk_idx in enumerate(c_indexes)
                }
            for future in cf.as_completed(future_ss):
                m = future_ss[future]
                try:
                    res = future.result()
                    if res.shape[0] != 0:
                        results.append(res)
                except ValueError:
                    pass
                except Exception as e:
                    print("Chunk {} worker died: ".format(m), e)
        return results

    def _base_search(
        self,
        query_string,
        threshold,
        search_func,
        chunk_size,
        s_index,
        on_disk,
        executor,
        n_workers,
    ):
        if on_disk:
            with tb.open_file(self.fp_filename, mode="r") as fp_file:
                count_ranges = fp_file.root.config[3]
        else:
            count_ranges = self.fps.count_ranges

        if s_index == "tanimoto":
            empty_np = np.ndarray((0,), dtype=[("mol_id", "<u4"), ("coeff", "<f4")])
        else:
            empty_np = np.ndarray((0,), dtype="<u4")

        query, fp_range = self._load_query_and_fp_range(
            query_string, count_ranges, threshold, s_index
        )
        if fp_range:
            if n_workers == 1 and not on_disk:
                np_res = search_func(
                    query, self.fps.fps, threshold, fp_range[0], fp_range[1]
                )
            else:
                results = self._parallel_run(
                    query,
                    search_func,
                    executor,
                    fp_range,
                    n_workers,
                    threshold,
                    chunk_size,
                    s_index,
                    on_disk,
                )
                if results:
                    np_res = np.concatenate(results)
                    if s_index != "substructure":
                        sort_results(np_res)
                else:
                    np_res = empty_np
        else:
            np_res = empty_np
        return np_res

import concurrent.futures as cf
import multiprocessing
from .io.chem import load_molecule, rdmol_to_efp, get_bounds_range
from .io.backends import PyTablesStorageBackend
from typing import Callable, Any, Tuple, Union
from .FPSim2lib import (
    _similarity_search,
    _substructure_search,
    sort_results,
    py_popcount,
)
from scipy import sparse
import numpy as np

SEARCH_TYPES = {"tanimoto": 0, "tversky": 1, "substructure": 2}


def on_disk_search(
    query: np.array,
    storage: Any,
    threshold: float,
    a: float,
    b: float,
    st: str,
    chunk_range: Tuple[int, int],
) -> np.ndarray:
    """Run a on disk search.

        Args:
            query: Preprocessed NumPy query array.
            fp_filename: Flat to load into memory or not the fps.
            threshold: Search threshold.
            a: alpha.
            b: beta.
            st: Type of search.
            chunk_range: List with the start and the end of the chunk to search.
        Returns:
            Numpy array with results.
    """
    fps = storage.get_fps_chunk(chunk_range)
    num_fields = len(fps[0])
    fps = fps.view("<u8")
    fps = fps.reshape(int(fps.size / num_fields), num_fields)
    if st == 2:
        res = _substructure_search(query, fps, threshold, 0, 0, st, 0, fps.shape[0])
    else:
        res = _similarity_search(query, fps, threshold, a, b, st, 0, fps.shape[0])
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
    storage = None

    def __init__(
        self,
        fp_filename: str,
        storage_backend: str = "pytables",
        in_memory_fps: bool = True,
        fps_sort: bool = False,
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

    def similarity(
        self, query_string: str, threshold: float, n_workers=1
    ) -> np.ndarray:
        """Run a tanimoto search

        Args:
            query_string: SMILES, InChi or molblock.
            threshold: Similarities with a coeff above the threshold will be kept.
            n_workers: Number of threads used for the search.
        Raises:
            Exception: If fps are not loaded into memory.
        Returns:
            Numpy array with ids and similarities.
        """
        if self.fps is None:
            raise Exception(
                "Load the fingerprints into memory before running a in memory search"
            )
        return self._base_search(
            query=query_string,
            threshold=threshold,
            a=0,
            b=0,
            search_func=_similarity_search,
            chunk_size=0,
            search_type="tanimoto",
            on_disk=False,
            executor=cf.ThreadPoolExecutor,
            n_workers=n_workers,
        )[["mol_id", "coeff"]]

    def on_disk_similarity(
        self,
        query_string: str,
        threshold: float,
        n_workers: int = 1,
        chunk_size: int = 250000,
    ) -> np.ndarray:
        """Run a on disk tanimoto search.

        Args:
            query_string: SMILES, InChi or molblock.
            threshold: Similarities with a coeff above the threshold will be kept.
            n_workers: Number of threads used for the search.
            chunk_size: Chunk size.
        Returns:
            Numpy array with ids and similarities.
        """
        return self._base_search(
            query=query_string,
            threshold=threshold,
            a=0,
            b=0,
            search_func=on_disk_search,
            chunk_size=chunk_size,
            search_type="tanimoto",
            on_disk=True,
            executor=cf.ProcessPoolExecutor,
            n_workers=n_workers,
        )[["mol_id", "coeff"]]

    def tversky(
        self,
        query_string: str,
        threshold: float,
        a: float,
        b: float,
        n_workers: int = 1,
    ) -> np.ndarray:
        """Run a tversky search

        Args:
            query_string: SMILES, InChi or molblock.
            threshold: Similarities with a coeff above the threshold will be kept.
            a: alpha
            b: beta
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
            query=query_string,
            threshold=threshold,
            a=a,
            b=b,
            search_func=_similarity_search,
            chunk_size=0,
            search_type="tversky",
            on_disk=False,
            executor=cf.ThreadPoolExecutor,
            n_workers=n_workers,
        )[["mol_id", "coeff"]]

    def on_disk_tversky(
        self,
        query_string: str,
        threshold: float,
        a: float,
        b: float,
        n_workers: int = 1,
        chunk_size: int = 250000,
    ) -> np.ndarray:
        """Run a on disk tversky search.

        Args:
            query_string: SMILES, InChi or molblock.
            threshold: Similarities with a coeff above the threshold will be kept.
            a: alpha
            b: beta
            n_workers: Number of threads used for the search.
            chunk_size: Chunk size.
        Returns:
            Numpy array with ids and similarities.
        """
        return self._base_search(
            query=query_string,
            threshold=threshold,
            a=a,
            b=b,
            search_func=on_disk_search,
            chunk_size=chunk_size,
            search_type="tversky",
            on_disk=True,
            executor=cf.ProcessPoolExecutor,
            n_workers=n_workers,
        )[["mol_id", "coeff"]]

    def substructure(self, query_string: str, n_workers: int = 1) -> np.ndarray:
        """Run a substructure screenout using an optimised calculation of tversky wiht a=1, b=0

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
            query=query_string,
            threshold=1.0,
            a=0,
            b=0,
            search_func=_substructure_search,
            chunk_size=0,
            search_type="substructure",
            on_disk=False,
            executor=cf.ThreadPoolExecutor,
            n_workers=n_workers,
        )

    def on_disk_substructure(
        self, query_string: str, n_workers: int = 1, chunk_size: int = 250000
    ) -> np.ndarray:
        """Run a on disk substructure screenout.

        Args:
            query_string: SMILES, InChi or molblock.
            n_workers: Number of processes used for the search.
            chunk_size: Chunk size.
        Returns:
            NumPy array with ids.
        """
        return self._base_search(
            query=query_string,
            threshold=1.0,
            a=0,
            b=0,
            search_func=on_disk_search,
            chunk_size=chunk_size,
            search_type="substructure",
            on_disk=True,
            executor=cf.ProcessPoolExecutor,
            n_workers=n_workers,
        )

    def _parallel_run(
        self,
        query: np.ndarray,
        search_func: Callable[..., np.ndarray],
        executor: Union[Callable[..., Any], None],
        fp_range: Union[Tuple[int, int], list],
        n_workers: int,
        threshold: float,
        a: float,
        b: float,
        search_type: str,
        chunk_size: int,
        on_disk: bool,
    ) -> np.ndarray:
        i_start = fp_range[0]
        i_end = fp_range[1]
        results = []
        with executor(max_workers=n_workers) as exe:
            if not on_disk:
                chunk_size = int((i_end - i_start) / n_workers)
                chunk_size = 1 if chunk_size == 0 else chunk_size
            chunks_ranges = [
                [x, x + chunk_size] for x in range(i_start, i_end, chunk_size)
            ]
            chunks_ranges[-1][1] = i_end
            if on_disk:
                future_ss = {
                    exe.submit(
                        search_func,
                        query,
                        self.storage,
                        threshold,
                        a,
                        b,
                        SEARCH_TYPES[search_type],
                        cr,
                    ): cr_id
                    for cr_id, cr in enumerate(chunks_ranges)
                }
            else:
                future_ss = {
                    exe.submit(
                        search_func,
                        query,
                        self.fps,
                        threshold,
                        a,
                        b,
                        SEARCH_TYPES[search_type],
                        cr[0],
                        cr[1],
                    ): cr_id
                    for cr_id, cr in enumerate(chunks_ranges)
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
        query: Union[np.uint32, str],
        threshold: float,
        a: float,
        b: float,
        search_func: Callable[..., np.ndarray],
        chunk_size: int,
        search_type: str,
        on_disk: bool,
        executor: Union[Callable[..., np.ndarray], None],
        n_workers: int,
    ) -> np.ndarray:
        if search_type == "substructure":
            empty_np = np.ndarray((0,), dtype="<u4")
        else:
            empty_np = np.ndarray((0,), dtype=[("mol_id", "<u4"), ("coeff", "<f4")])

        if isinstance(query, np.uint32):
            np_query = self.fps[query]
            # +1 to avoid the diagonal of the matrix
            start_idx = query + 1
        else:
            np_query = self.load_query(query)
            start_idx = None

        fp_range = get_bounds_range(
            np_query, threshold, a, b, self.popcnt_bins, search_type
        )

        if fp_range:
            if n_workers == 1:
                if on_disk:
                    np_res = search_func(
                        np_query,
                        self.storage,
                        threshold,
                        a,
                        b,
                        SEARCH_TYPES[search_type],
                        fp_range,
                    )
                else:
                    np_res = search_func(
                        np_query,
                        self.fps,
                        threshold,
                        a,
                        b,
                        SEARCH_TYPES[search_type],
                        start_idx
                        if start_idx and fp_range[0] < start_idx
                        else fp_range[0],
                        fp_range[1],
                    )
            else:
                results = self._parallel_run(
                    np_query,
                    search_func,
                    executor,
                    fp_range,
                    n_workers,
                    threshold,
                    a,
                    b,
                    search_type,
                    chunk_size,
                    on_disk,
                )
                if results:
                    np_res = np.concatenate(results)
                    if search_type != "substructure":
                        sort_results(np_res)
                else:
                    np_res = empty_np
        else:
            np_res = empty_np
        return np_res

    def symmetric_distance_matrix(
        self,
        threshold: float,
        search_type: str = "tanimoto",
        a: float = 0,
        b: float = 0,
        n_workers: int = multiprocessing.cpu_count(),
    ) -> sparse.csr.csr_matrix:
        """ Returns a CSR distance matrix """

        if search_type == "tversky" and a != b:
            raise Exception("tversky with a != b is asymmetric")

        from tqdm import tqdm

        # shuffle indexes so we can estimate how long can a run take
        idxs = np.arange(self.fps.shape[0], dtype=np.uint32)
        np.random.shuffle(idxs)

        rows = []
        cols = []
        data = []
        if n_workers == 1:
            for idx in tqdm(idxs, total=idxs.shape[0]):
                np_res = self._base_search(
                    query=idx,
                    threshold=threshold,
                    a=a,
                    b=b,
                    search_func=_similarity_search,
                    chunk_size=0,
                    search_type=search_type,
                    on_disk=False,
                    executor=None,
                    n_workers=1,
                )
                for r in np_res:
                    rows.append(idx)
                    cols.append(r["idx"])
                    data.append(r["coeff"])
                    # symmetry
                    rows.append(r["idx"])
                    cols.append(idx)
                    data.append(r["coeff"])

        else:
            with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        self._base_search,
                        query=idx,
                        threshold=threshold,
                        a=a,
                        b=b,
                        search_func=_similarity_search,
                        chunk_size=0,
                        search_type=search_type,
                        on_disk=False,
                        executor=None,
                        n_workers=1,
                    ): idx
                    for idx in idxs
                }
                for future in tqdm(cf.as_completed(future_to_idx), total=idxs.shape[0]):
                    idx = future_to_idx[future]
                    np_res = future.result()
                    for r in np_res:
                        rows.append(idx)
                        cols.append(r["idx"])
                        data.append(r["coeff"])
                        # symmetry
                        rows.append(r["idx"])
                        cols.append(idx)
                        data.append(r["coeff"])

        csr_matrix = sparse.csr_matrix(
            (data, (rows, cols)), shape=(self.fps.shape[0], self.fps.shape[0])
        )

        # similarity to distance
        csr_matrix.data = 1 - csr_matrix.data
        return csr_matrix

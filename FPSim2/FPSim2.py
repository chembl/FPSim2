import concurrent.futures as cf
from rdkit.DataStructs import ExplicitBitVect
from .io.chem import get_bounds_range
from typing import Callable, Any, Tuple, Union
from .FPSim2lib import (
    TanimotoSearch,
    TverskySearch,
    SubstructureScreenout,
    TanimotoSearchTopK,
)
from .FPSim2lib.utils import SortResults
from scipy.sparse import csr_matrix
from .base import BaseEngine
import numpy as np


def on_disk_search(
    search_func: str, query: np.array, storage: Any, args, chunk: Tuple[int, int],
) -> np.ndarray:
    fps = storage.get_fps_chunk(chunk)
    num_fields = len(fps[0])
    fps = fps.view("<u8")
    fps = fps.reshape(int(fps.size / num_fields), num_fields)
    res = globals()[search_func](query, fps, *args, 0, fps.shape[0])
    return res


class FPSim2Engine(BaseEngine):
    """FPSim2 class to run fast CPU searches.

    Parameters
    ----------
    fp_filename : str
        Fingerprints database file path.

    in_memory_fps : bool
        Whether if the FPs should be loaded into memory or not.

    fps_sort : bool
        Whether if the FPs should be sorted by popcnt after being loaded into memory or not.

    storage_backend : str
        Storage backend to use (only pytables available at the moment).

    Attributes
    ----------
    fps : Numpy array
        Fingerprints.

    popcnt_bins : list
        List with the popcount bins ranges.

    fp_type : str
        Fingerprint type used to create the fingerprints.

    fp_params : dict
        Parameters used to create the fingerprints.

    rdkit_ver : dict
        RDKit version used to create the fingerprints.

    Examples
    --------
    """

    def __init__(
        self,
        fp_filename: str = "",
        in_memory_fps: bool = True,
        fps_sort: bool = False,
        storage_backend: str = "pytables",
        conn_url: str = "",
        table_name: str = "",
        pg_schema: str = ""
    ) -> None:
        super(FPSim2Engine, self).__init__(
            fp_filename=fp_filename,
            storage_backend=storage_backend,
            in_memory_fps=in_memory_fps,
            fps_sort=fps_sort,
            conn_url=conn_url,
            table_name=table_name,
            pg_schema=pg_schema
        )
        self.empty_sim = np.ndarray((0,), dtype=[("mol_id", "<u4"), ("coeff", "<f4")])
        self.empty_subs = np.ndarray((0,), dtype="<u4")

    def similarity(
        self, query: Union[str, ExplicitBitVect], threshold: float, n_workers=1
    ) -> np.ndarray:
        """Runs a Tanimoto search.

        Parameters
        ----------
        query : Union[str, ExplicitBitVect]
            SMILES, InChI, molblock or fingerprint as ExplicitBitVect.

        threshold: float
            Similarity threshold.

        n_workers : int
            Number of threads used for the search.

        Returns
        -------
        results : numpy array
            Similarity results.
        """
        if self.fps is None:
            raise Exception(
                "Load the fingerprints into memory before running a in memory search"
            )

        query = self.load_query(query)
        bounds = get_bounds_range(
            query, threshold, None, None, self.popcnt_bins, "tanimoto"
        )

        if not bounds:
            results = self.empty_sim
        else:
            if n_workers == 1:
                results = TanimotoSearch(query, self.fps, threshold, *bounds)
            else:
                results = self._parallel(
                    search_func=TanimotoSearch,
                    executor=cf.ThreadPoolExecutor,
                    query=query,
                    db=self.fps,
                    args=(threshold,),
                    bounds=bounds,
                    on_disk=False,
                    n_workers=n_workers,
                    chunk_size=0,
                )
        return results[["mol_id", "coeff"]]

    def on_disk_similarity(
        self,
        query_string: str,
        threshold: float,
        n_workers: int = 1,
        chunk_size: int = 0,
    ) -> np.ndarray:
        """Runs a on disk Tanimoto search.

        Parameters
        ----------
        query_string : str
            SMILES, InChI or molblock.

        threshold: float
            Similarity threshold.

        n_workers : int
            Number of processes used for the search.

        chunk_size : float
            Chunk size.

        Returns
        -------
        results : numpy array
            Similarity results.
        """
        if not chunk_size:
            chunk_size = self.storage.chunk_size

        np_query = self.load_query(query_string)
        bounds = get_bounds_range(
            np_query, threshold, None, None, self.popcnt_bins, "tanimoto"
        )
        if not bounds:
            results = self.empty_sim
        else:
            if n_workers == 1:
                results = self._on_disk_single_core(
                    np_query, (threshold,), bounds, chunk_size, TanimotoSearch
                )
            else:
                results = self._parallel(
                    search_func=TanimotoSearch,
                    executor=cf.ProcessPoolExecutor,
                    query=np_query,
                    db=self.storage,
                    args=(threshold,),
                    bounds=bounds,
                    on_disk=True,
                    n_workers=n_workers,
                    chunk_size=chunk_size,
                )
        return results[["mol_id", "coeff"]]

    def tversky(
        self,
        query_string: str,
        threshold: float,
        a: float,
        b: float,
        n_workers: int = 1,
    ) -> np.ndarray:
        """Runs a Tversky search.

        Parameters
        ----------
        query_string : str
            SMILES, InChI or molblock.

        threshold: float
            Similarity threshold.

        a: float
            alpha

        b: float
            beta

        n_workers : int
            Number of threads used for the search.

        chunk_size : float
            Chunk size.

        Returns
        -------
        results : numpy array
            Similarity results.
        """
        if self.fps is None:
            raise Exception(
                "Load the fingerprints into memory before running a in memory search"
            )

        np_query = self.load_query(query_string)
        bounds = get_bounds_range(
            np_query, threshold, a, b, self.popcnt_bins, "tversky"
        )
        if not bounds:
            results = self.empty_sim
        else:
            if n_workers == 1:
                results = TverskySearch(np_query, self.fps, threshold, a, b, *bounds)
            else:
                results = self._parallel(
                    search_func=TverskySearch,
                    executor=cf.ThreadPoolExecutor,
                    query=np_query,
                    db=self.fps,
                    args=(threshold, a, b),
                    bounds=bounds,
                    on_disk=False,
                    n_workers=n_workers,
                    chunk_size=0,
                )
        return results[["mol_id", "coeff"]]

    def on_disk_tversky(
        self,
        query_string: str,
        threshold: float,
        a: float,
        b: float,
        n_workers: int = 1,
        chunk_size: int = None,
    ) -> np.ndarray:
        """Runs a on disk Tversky search.

        Parameters
        ----------
        query_string : str
            SMILES, InChI or molblock.

        threshold: float
            Similarity threshold.

        a: float
            alpha

        b: float
            beta

        n_workers : int
            Number of processes used for the search.

        chunk_size : float
            Chunk size.

        Returns
        -------
        results : numpy array
            Similarity results.
        """
        if not chunk_size:
            chunk_size = self.storage.chunk_size

        np_query = self.load_query(query_string)
        bounds = get_bounds_range(
            np_query, threshold, a, b, self.popcnt_bins, "tversky"
        )
        if not bounds:
            results = self.empty_sim
        else:
            if n_workers == 1:
                results = self._on_disk_single_core(
                    np_query, (threshold, a, b), bounds, chunk_size, TverskySearch
                )
            else:
                results = self._parallel(
                    search_func=TverskySearch,
                    executor=cf.ProcessPoolExecutor,
                    query=np_query,
                    db=self.storage,
                    args=(threshold, a, b),
                    bounds=bounds,
                    on_disk=True,
                    n_workers=n_workers,
                    chunk_size=chunk_size,
                )
        return results[["mol_id", "coeff"]]

    def substructure(self, query_string: str, n_workers: int = 1) -> np.ndarray:
        """Run a substructure screenout using an optimised calculation of tversky wiht a=1, b=0

        Parameters
        ----------
        query_string : str
            SMILES, InChI or molblock.

        n_workers : int
            Number of processes used for the search.

        chunk_size : float
            Chunk size.

        Returns
        -------
        results : numpy array
            Substructure results.
        """
        if self.fps is None:
            raise Exception(
                "Load the fingerprints into memory before running a in memory search"
            )
        np_query = self.load_query(query_string)
        bounds = get_bounds_range(
            np_query, 1, None, None, self.popcnt_bins, "substructure"
        )
        if not bounds:
            results = self.empty_subs
        else:
            if n_workers == 1:
                results = SubstructureScreenout(np_query, self.fps, *bounds)
            else:
                results = self._parallel(
                    search_func=SubstructureScreenout,
                    executor=cf.ThreadPoolExecutor,
                    query=np_query,
                    db=self.fps,
                    args=(),
                    bounds=bounds,
                    on_disk=False,
                    n_workers=n_workers,
                    chunk_size=0,
                )
        return results

    def on_disk_substructure(
        self, query_string: str, n_workers: int = 1, chunk_size: int = None
    ) -> np.ndarray:
        """Run a on disk substructure screenout.

        Parameters
        ----------
        query_string : str
            SMILES, InChI or molblock.

        n_workers : int
            Number of processes used for the search.

        chunk_size : float
            Chunk size.

        Returns
        -------
        results : numpy array
            Substructure results.
        """
        if not chunk_size:
            chunk_size = self.storage.chunk_size

        np_query = self.load_query(query_string)
        bounds = get_bounds_range(
            np_query, 1, None, None, self.popcnt_bins, "substructure"
        )
        if not bounds:
            results = self.empty_subs
        else:
            if n_workers == 1:
                results = self._on_disk_single_core(
                    np_query, (), bounds, chunk_size, SubstructureScreenout
                )
            else:
                results = self._parallel(
                    search_func=SubstructureScreenout,
                    executor=cf.ProcessPoolExecutor,
                    query=np_query,
                    db=self.storage,
                    args=(),
                    bounds=bounds,
                    on_disk=True,
                    n_workers=n_workers,
                    chunk_size=chunk_size,
                )
        return results

    def _on_disk_single_core(
        self,
        query: np.array,
        args: Any,
        bounds: Tuple[int, int],
        chunk_size: int,
        search_func: Callable[..., np.ndarray],
    ) -> np.ndarray:
        chunks = ((x, x + chunk_size) for x in range(*bounds, chunk_size))
        results = []
        for chunk in chunks:
            res = on_disk_search(search_func.__name__, query, self.storage, args, chunk)
            if len(res) > 0:
                results.append(res)
        if len(results):
            results = np.concatenate(results)
            if not search_func.__name__ == "SubstructureScreenout":
                SortResults(results)
        else:
            if not search_func.__name__ == "SubstructureScreenout":
                results = self.empty_sim
            else:
                results = self.empty_subs
        return results

    def _parallel(
        self,
        search_func: Callable[..., np.ndarray],
        executor: Callable[..., Any],
        query: np.array,
        db: np.array,
        args: Tuple,
        bounds: Tuple[int, int],
        on_disk: bool,
        n_workers: int,
        chunk_size: Union[int],
    ) -> np.ndarray:
        start, end = bounds
        results = []
        with executor(max_workers=n_workers) as exe:
            if not on_disk:
                chunk_size = (end - start) // n_workers
                chunk_size = 1 if chunk_size == 0 else chunk_size
            chunks = [[x, x + chunk_size] for x in range(start, end, chunk_size)]
            chunks[-1][1] = end
            if on_disk:
                future_ss = {
                    exe.submit(
                        on_disk_search,
                        search_func.__name__,  # PyCapsule is not pickable...
                        query,
                        db,
                        args,
                        chunk,
                    ): chunk_id
                    for chunk_id, chunk in enumerate(chunks)
                }
            else:
                future_ss = {
                    exe.submit(search_func, query, db, *args, *chunk): chunk_id
                    for chunk_id, chunk in enumerate(chunks)
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
                    raise Exception("Chunk {} worker died: ".format(m), e)

        if results:
            results = np.concatenate(results)
            if not search_func.__name__ == "SubstructureScreenout":
                SortResults(results)
        else:
            if not search_func.__name__ == "SubstructureScreenout":
                results = self.empty_sim
            else:
                results = self.empty_subs
        return results

    def symmetric_distance_matrix(
        self,
        threshold: float,
        search_type: str = "tanimoto",
        a: float = 0,
        b: float = 0,
        n_workers: int = 4,
    ) -> csr_matrix:
        """Computes the Tanimoto similarity matrix of the set.

        Parameters
        ----------
        threshold : float
            Similarity threshold.

        search_type : str
            Type of search.

        a : float
            alpha in Tversky search.

        b : float
            beta in Tversky search.

        n_workers : int
            Number of threads to use.

        Returns
        -------
        results : numpy array
            Similarity results.
        """
        if search_type == "tversky":
            if a != b:
                raise Exception("tversky with a != b is asymmetric")
            search_func = TverskySearch
            args = (threshold, a, b)
        else:
            search_func = TanimotoSearch
            args = (threshold,)

        from tqdm import tqdm

        # shuffle indexes so we can estimate how long can a run take
        idxs = np.arange(self.fps.shape[0], dtype=np.uint32)
        np.random.shuffle(idxs)

        def run(idx):
            np_query = self.fps[idx]
            bounds = get_bounds_range(
                np_query, threshold, a, b, self.popcnt_bins, search_type
            )
            sym_bounds = (max(idx + 1, bounds[0]), bounds[1])
            return search_func(np_query, self.fps, *args, *sym_bounds)

        rows = []
        cols = []
        data = []
        if n_workers == 1:
            for idx in tqdm(idxs, total=idxs.shape[0]):
                np_res = run(idx)
                for r in np_res:
                    rows.append(idx)
                    cols.append(r["idx"])
                    data.append(r["coeff"])
        else:
            with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {executor.submit(run, idx,): idx for idx in idxs}
                for future in tqdm(cf.as_completed(future_to_idx), total=idxs.shape[0]):
                    idx = future_to_idx[future]
                    np_res = future.result()
                    for r in np_res:
                        rows.append(idx)
                        cols.append(r["idx"])
                        data.append(r["coeff"])

        sparse_matrix = csr_matrix(
            (data + data, (rows + cols, cols + rows)),
            shape=(self.fps.shape[0], self.fps.shape[0]),
        )

        # similarity to distance
        sparse_matrix.data = 1 - sparse_matrix.data
        return sparse_matrix

    def top_k(
        self, query: Union[str, ExplicitBitVect], k: int, threshold: float, n_workers=1
    ) -> np.ndarray:
        """Runs a Tanimoto top-K search.

        Parameters
        ----------
        query : Union[str, ExplicitBitVect]
            SMILES, InChI, molblock or fingerprint as ExplicitBitVect.

        threshold: float
            Similarity threshold.

        n_workers : int
            Number of threads used for the search.

        Returns
        -------
        results : numpy array
            Similarity results.
        """
        if self.fps is None:
            raise Exception(
                "Load the fingerprints into memory before running a in memory search"
            )

        query = self.load_query(query)
        bounds = get_bounds_range(
            query, threshold, None, None, self.popcnt_bins, "tanimoto"
        )

        if not bounds:
            results = self.empty_sim
        else:
            if n_workers == 1:
                results = TanimotoSearchTopK(query, self.fps, k, threshold, *bounds)
            else:
                results = self._parallel(
                    search_func=TanimotoSearchTopK,
                    executor=cf.ThreadPoolExecutor,
                    query=query,
                    db=self.fps,
                    args=(k, threshold,),
                    bounds=bounds,
                    on_disk=False,
                    n_workers=n_workers,
                    chunk_size=0,
                )
        return results[['mol_id', 'coeff']][:k]

import concurrent.futures as cf
from rdkit.DataStructs import ExplicitBitVect
from .io.chem import get_bounds_range
from typing import Callable, Any, Tuple, Union
from .FPSim2lib import (
    TanimotoSearch,
    TverskySearch,
    SubstructureScreenout,
    TanimotoSearchTopK,
    DiceSearch,
    CosineSearch,
)
from .FPSim2lib.utils import SortResults
from scipy.sparse import csr_matrix
from .base import BaseEngine
import numpy as np


def on_disk_search(
    search_func: str,
    query: np.array,
    storage: Any,
    args,
    chunk: Tuple[int, int],
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
        pg_schema: str = "",
    ) -> None:
        super(FPSim2Engine, self).__init__(
            fp_filename=fp_filename,
            storage_backend=storage_backend,
            in_memory_fps=in_memory_fps,
            fps_sort=fps_sort,
            conn_url=conn_url,
            table_name=table_name,
            pg_schema=pg_schema,
        )
        self.empty_sim = np.ndarray((0,), dtype=[("mol_id", "<u4"), ("coeff", "<f4")])
        self.empty_subs = np.ndarray((0,), dtype="<u4")

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

    def _search(
        self,
        query: Union[str, ExplicitBitVect],
        search_type: str = "tanimoto",
        threshold: float = None,
        a: float = None,
        b: float = None,
        k: int = None,
        n_workers: int = 1,
        on_disk: bool = False,
        chunk_size: int = None,
    ) -> np.ndarray:
        """Generic search function supporting tanimoto, tversky, substructure, cosine, and dice searches.

        Parameters
        ----------
        query : Union[str, ExplicitBitVect]
            SMILES, InChI, molblock or fingerprint as ExplicitBitVect.
        search_type : str
            Type of search ('tanimoto', 'tversky', 'substructure', 'top_k', 'cosine', 'dice')
        threshold : float
            Similarity threshold.
        a : float
            Alpha parameter for Tversky search.
        b : float
            Beta parameter for Tversky search.
        k : int
            Number of top results to return (for top_k search).
        n_workers : int
            Number of threads/processes used for the search.
        on_disk : bool
            Whether to perform on-disk search.
        chunk_size : int
            Chunk size for on-disk searches.

        Returns
        -------
        results : numpy array
            Search results.
        """
        if not on_disk and self.fps is None:
            raise Exception(
                "Load the fingerprints into memory before running a in memory search"
            )

        if on_disk and not chunk_size:
            chunk_size = self.storage.chunk_size

        # Map search types to their corresponding functions and parameters
        search_funcs = {
            "tanimoto": (TanimotoSearch, (threshold,)),
            "tversky": (TverskySearch, (threshold, a, b)),
            "substructure": (SubstructureScreenout, ()),
            "top_k": (TanimotoSearchTopK, (k, threshold)),
            "dice": (DiceSearch, (threshold,)),
            "cosine": (CosineSearch, (threshold,)),
        }

        np_query = self.load_query(query)
        search_func, args = search_funcs[search_type]

        # Get bounds for the search
        threshold_val = 1 if search_type == "substructure" else threshold
        bounds = get_bounds_range(
            np_query, threshold_val, a, b, self.popcnt_bins, search_type
        )

        if not bounds:
            return self.empty_subs if search_type == "substructure" else self.empty_sim

        if n_workers == 1:
            if on_disk:
                results = self._on_disk_single_core(
                    np_query, args, bounds, chunk_size, search_func
                )
            else:
                results = search_func(np_query, self.fps, *args, *bounds)
        else:
            results = self._parallel(
                search_func=search_func,
                executor=cf.ProcessPoolExecutor if on_disk else cf.ThreadPoolExecutor,
                query=np_query,
                db=self.storage if on_disk else self.fps,
                args=args,
                bounds=bounds,
                on_disk=on_disk,
                n_workers=n_workers,
                chunk_size=chunk_size if on_disk else 0,
            )

        if search_type == "top_k":
            return results[["mol_id", "coeff"]][:k]
        return (
            results[["mol_id", "coeff"]] if search_type != "substructure" else results
        )

    def tanimoto(
        self, query: Union[str, ExplicitBitVect], threshold: float, n_workers=1
    ) -> np.ndarray:
        return self._search(query, "tanimoto", threshold=threshold, n_workers=n_workers)

    def on_disk_tanimoto(
        self,
        query: Union[str, ExplicitBitVect],
        threshold: float,
        n_workers: int = 1,
        chunk_size: int = 0,
    ) -> np.ndarray:
        return self._search(
            query,
            "tanimoto",
            threshold=threshold,
            n_workers=n_workers,
            on_disk=True,
            chunk_size=chunk_size,
        )

    def cosine(
        self, query: Union[str, ExplicitBitVect], threshold: float, n_workers=1
    ) -> np.ndarray:
        return self._search(query, "cosine", threshold=threshold, n_workers=n_workers)

    def on_disk_cosine(
        self,
        query: Union[str, ExplicitBitVect],
        threshold: float,
        n_workers: int = 1,
        chunk_size: int = 0,
    ) -> np.ndarray:
        return self._search(
            query,
            "cosine",
            threshold=threshold,
            n_workers=n_workers,
            on_disk=True,
            chunk_size=chunk_size,
        )

    def dice(
        self, query: Union[str, ExplicitBitVect], threshold: float, n_workers=1
    ) -> np.ndarray:
        return self._search(query, "dice", threshold=threshold, n_workers=n_workers)

    def on_disk_dice(
        self,
        query: Union[str, ExplicitBitVect],
        threshold: float,
        n_workers: int = 1,
        chunk_size: int = 0,
    ) -> np.ndarray:
        return self._search(
            query,
            "dice",
            threshold=threshold,
            n_workers=n_workers,
            on_disk=True,
            chunk_size=chunk_size,
        )

    def similarity(
        self, query: Union[str, ExplicitBitVect], threshold: float, n_workers=1
    ) -> np.ndarray:
        return self._search(query, "tanimoto", threshold=threshold, n_workers=n_workers)

    def on_disk_similarity(
        self,
        query_string: str,
        threshold: float,
        n_workers: int = 1,
        chunk_size: int = 0,
    ) -> np.ndarray:
        return self._search(
            query_string,
            "tanimoto",
            threshold=threshold,
            n_workers=n_workers,
            on_disk=True,
            chunk_size=chunk_size,
        )

    def tversky(
        self,
        query_string: str,
        threshold: float,
        a: float,
        b: float,
        n_workers: int = 1,
    ) -> np.ndarray:
        return self._search(
            query_string, "tversky", threshold=threshold, a=a, b=b, n_workers=n_workers
        )

    def on_disk_tversky(
        self,
        query_string: str,
        threshold: float,
        a: float,
        b: float,
        n_workers: int = 1,
        chunk_size: int = None,
    ) -> np.ndarray:
        return self._search(
            query_string,
            "tversky",
            threshold=threshold,
            a=a,
            b=b,
            n_workers=n_workers,
            on_disk=True,
            chunk_size=chunk_size,
        )

    def substructure(self, query_string: str, n_workers: int = 1) -> np.ndarray:
        return self._search(query_string, "substructure", n_workers=n_workers)

    def on_disk_substructure(
        self, query_string: str, n_workers: int = 1, chunk_size: int = None
    ) -> np.ndarray:
        return self._search(
            query_string,
            "substructure",
            n_workers=n_workers,
            on_disk=True,
            chunk_size=chunk_size,
        )

    def top_k(
        self, query: Union[str, ExplicitBitVect], k: int, threshold: float, n_workers=1
    ) -> np.ndarray:
        return self._search(
            query, "top_k", threshold=threshold, k=k, n_workers=n_workers
        )

    def on_disk_top_k(
        self,
        query: Union[str, ExplicitBitVect],
        k: int,
        threshold: float,
        n_workers: int = 1,
        chunk_size: int = None,
    ) -> np.ndarray:
        return self._search(
            query,
            "top_k",
            threshold=threshold,
            k=k,
            n_workers=n_workers,
            on_disk=True,
            chunk_size=chunk_size,
        )

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
                future_to_idx = {
                    executor.submit(
                        run,
                        idx,
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

        sparse_matrix = csr_matrix(
            (data + data, (rows + cols, cols + rows)),
            shape=(self.fps.shape[0], self.fps.shape[0]),
        )

        # similarity to distance
        sparse_matrix.data = 1 - sparse_matrix.data
        return sparse_matrix

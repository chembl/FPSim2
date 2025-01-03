import concurrent.futures as cf
from rdkit.DataStructs import ExplicitBitVect
from .io.chem import get_bounds_range, METRICS
from typing import Callable, Any, Tuple, Union
from .FPSim2lib import (
    GenericSearch,
    TverskySearch,
    SubstructureScreenout,
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
    return globals()[search_func](query, fps, *args, 0, fps.shape[0])


class FPSim2Engine(BaseEngine):
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

    def __str__(self):
        return super().__str__()

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
                        search_func.__name__,
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
        search_type: str = "similarity",
        metric: str = "tanimoto",
        threshold: float = None,
        a: float = None,
        b: float = None,
        k: int = 0,
        n_workers: int = 1,
        on_disk: bool = False,
        chunk_size: int = None,
    ) -> np.ndarray:
        if not on_disk and self.fps is None:
            raise Exception(
                "Load the fingerprints into memory before running a in memory search"
            )
        if on_disk and not chunk_size:
            chunk_size = self.storage.chunk_size

        if metric not in METRICS:
            raise ValueError(
                f"Invalid metric: {metric}. Available metrics are: {list(METRICS.keys())}"
            )
        calc_type = METRICS[metric]

        search_funcs = {
            "similarity": (GenericSearch, (threshold, k, calc_type)),
            "tversky": (TverskySearch, (threshold, a, b)),
            "substructure": (SubstructureScreenout, ()),
        }

        np_query = self.load_query(query)
        search_func, args = search_funcs[search_type]

        # Get bounds
        threshold_val = 1 if search_type == "substructure" else threshold
        if search_type in ("tversky", "substructure"):
            bounds_par = search_type
        else:
            bounds_par = metric
        bounds = get_bounds_range(
            np_query, threshold_val, a, b, self.popcnt_bins, bounds_par
        )
        if not bounds:
            return self.empty_subs if search_type == "substructure" else self.empty_sim

        # Dispatch
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

    def similarity(
        self,
        query: Union[str, ExplicitBitVect],
        threshold: float,
        metric: str = "tanimoto",
        n_workers=1,
    ) -> np.ndarray:
        return self._search(
            query,
            search_type="similarity",
            metric=metric,
            threshold=threshold,
            n_workers=n_workers,
        )

    def on_disk_similarity(
        self,
        query_string: str,
        threshold: float,
        metric: str = "tanimoto",
        n_workers: int = 1,
        chunk_size: int = 0,
    ) -> np.ndarray:
        return self._search(
            query_string,
            search_type="similarity",
            metric=metric,
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
        self,
        query: Union[str, ExplicitBitVect],
        k: int,
        threshold: float,
        metric="tanimoto",
        n_workers=1,
    ) -> np.ndarray:
        return self._search(
            query,
            "similarity",
            metric=metric,
            threshold=threshold,
            k=k,
            n_workers=n_workers,
        )

    def on_disk_top_k(
        self,
        query: Union[str, ExplicitBitVect],
        k: int,
        threshold: float,
        metric="tanimoto",
        n_workers=1,
        chunk_size: int = None,
    ) -> np.ndarray:
        return self._search(
            query,
            "similarity",
            metric=metric,
            threshold=threshold,
            k=k,
            n_workers=n_workers,
            on_disk=True,
            chunk_size=chunk_size,
        )

    def symmetric_distance_matrix(
        self,
        threshold: float,
        metric: str = "tanimoto",
        n_workers: int = 4,
    ) -> csr_matrix:
        search_func = GenericSearch

        if metric not in METRICS:
            raise ValueError(
                f"Invalid metric: {metric}. Available metrics are: {list(METRICS.keys())}"
            )
        calc_type = METRICS[metric]

        args = (threshold, 0, calc_type)

        from tqdm import tqdm

        idxs = np.arange(self.fps.shape[0], dtype=np.uint32)
        np.random.shuffle(idxs)

        def run(idx):
            np_query = self.fps[idx]
            bounds = get_bounds_range(
                np_query, threshold, 0, 0, self.popcnt_bins, metric
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
                future_to_idx = {executor.submit(run, idx): idx for idx in idxs}
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
        sparse_matrix.data = 1 - sparse_matrix.data
        return sparse_matrix

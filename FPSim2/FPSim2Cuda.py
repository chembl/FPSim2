from typing import Union
from rdkit.DataStructs import ExplicitBitVect
from rdkit import Chem
from .io.chem import get_bounds_range
from .base import BaseEngine
import numpy as np
import cupy as cp


class FPSim2CudaEngine(BaseEngine):
    """FPSim2 class to run fast GPU Tanimoto searches.

    Parameters
    ----------
    fp_filename : str
        Fingerprints database file path.

    fps_sort : bool
        Wheter if the FPs should be sorted after being loaded into memory or not.

    storage_backend : str
        Which storage backend to use.
    """

    raw_kernel = r"""
        extern "C" __global__
        void taniRAW(const unsigned long long int* query,
                     const unsigned long long int* qcount,
                     const unsigned long long int* db,
                     const unsigned long long int* popcnts,
                     const float* threshold,
                     float* out) {{

            // Shared block array. Only visible for threads in same block
            __shared__ int common[{block}];

            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            common[threadIdx.x] = __popcll(query[threadIdx.x] & db[tid]);

            // threads need to wait until all threads finish
            __syncthreads();

            // thread 0 in each block sums the common bits
            // and calcs the final coeff
            if(0 == threadIdx.x)
            {{
                int comm_sum = 0;
                for(int i=0; i<{block}; i++)
                    comm_sum += common[i];

                float coeff = 0.0;
                coeff = *qcount + popcnts[blockIdx.x] - comm_sum;
                if (coeff != 0.0)
                    coeff = comm_sum / coeff;
                out[blockIdx.x] = coeff >= *threshold ? coeff : 0.0;
            }}
        }}
    """

    def __init__(
        self,
        fp_filename: str = "",
        storage_backend: str = "pytables",
        conn_url: str = "",
        table_name: str = "",
        pg_schema: str = "",
    ) -> None:
        super(FPSim2CudaEngine, self).__init__(
            fp_filename=fp_filename,
            storage_backend=storage_backend,
            in_memory_fps=True,
            fps_sort=False,
            conn_url=conn_url,
            table_name=table_name,
            pg_schema=pg_schema,
        )
        # copy all the stuff to the GPU
        self.cuda_db = cp.asarray(self.fps[:, 1:-1])
        self.cuda_ids = cp.asarray(self.fps[:, 0])
        self.cuda_db_popcnts = cp.asarray(self.fps[:, -1])
        self.cupy_kernel = cp.RawKernel(
            self.raw_kernel.format(block=self.cuda_db.shape[1]),
            name="taniRAW",
            options=("-std=c++14",),
        )

    def _raw_kernel_search(self, np_query, threshold):
        # get the range of the molecule subset to screen
        fp_range = get_bounds_range(
            np_query, threshold, 0, 0, self.popcnt_bins, "tanimoto"
        )

        if not fp_range:
            return np.ndarray(0), np.ndarray(0)

        cuda_threshold = cp.asarray(threshold, dtype=cp.float32)
        cuda_query = cp.asarray(np_query[1:-1], dtype=cp.uint64)
        cuda_query_popcount = cp.asarray(np_query[-1], dtype=cp.uint64)

        slice_range = slice(*fp_range)
        grid_size = int(fp_range[1] - fp_range[0])
        sims = cp.zeros(grid_size, dtype=cp.float32)
        block_size = self.cuda_db.shape[1]  # number of uint64 elements per fingerprint

        # run the kernel
        self.cupy_kernel(
            (grid_size,),
            (block_size,),
            (
                cuda_query,
                cuda_query_popcount,
                self.cuda_db[slice_range],
                self.cuda_db_popcnts[slice_range],
                cuda_threshold,
                sims,
            ),
        )
        mask = cp.where(sims > 0)[0]
        return cp.asnumpy(self.cuda_ids[slice_range][mask]), cp.asnumpy(sims[mask])

    def similarity(
        self,
        query: Union[str, ExplicitBitVect, Chem.Mol],
        threshold: float,
        full_sanitization: bool = True,
    ) -> np.ndarray:
        """Runs a CUDA Tanimoto search.

        Parameters
        ----------
        query_string : str
            SMILES, InChI or molblock.

        threshold: float
            Similarity threshold.

        Returns
        -------
        results : numpy array
            Similarity results.
        """
        np_query = self.load_query(query, full_sanitization=full_sanitization)
        ids, sims = self._raw_kernel_search(np_query, threshold)

        # create results numpy array
        results = np.empty(
            len(ids),
            dtype=np.dtype([("mol_id", "u4"), ("coeff", "f4")]),
        )
        results["mol_id"] = ids
        results["coeff"] = sims
        results[::-1].sort(order="coeff")
        return results

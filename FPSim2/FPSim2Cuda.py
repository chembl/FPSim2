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
                     const int* num_mols,
                     const int* fp_size,
                     float* out) {{

            // One thread per molecule - more efficient than shared memory reduction
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            
            if (tid < *num_mols) {{
                int common_bits = 0;
                int fp_sz = *fp_size;
                
                // Compute popcount of (query AND db_fp) for each uint64 block
                for (int i = 0; i < fp_sz; i++) {{
                    common_bits += __popcll(query[i] & db[tid * fp_sz + i]);
                }}
                
                // Tanimoto = intersection / (A + B - intersection)
                float coeff = 0.0f;
                int union_count = *qcount + popcnts[tid] - common_bits;
                if (union_count > 0) {{
                    coeff = (float)common_bits / (float)union_count;
                }}
                
                out[tid] = coeff >= *threshold ? coeff : 0.0f;
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
        self.fp_size = self.cuda_db.shape[1]  # number of uint64 elements per fingerprint
        self.cupy_kernel = cp.RawKernel(
            self.raw_kernel,
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
        num_mols = int(fp_range[1] - fp_range[0])
        sims = cp.zeros(num_mols, dtype=cp.float32)
        
        # Flatten db slice for contiguous memory access
        db_slice = self.cuda_db[slice_range].ravel()
        popcnts_slice = self.cuda_db_popcnts[slice_range]
        
        # Grid/block configuration: 256 threads per block, one thread per molecule
        block_size = 256
        grid_size = (num_mols + block_size - 1) // block_size
        
        cuda_num_mols = cp.asarray(num_mols, dtype=cp.int32)
        cuda_fp_size = cp.asarray(self.fp_size, dtype=cp.int32)

        # run the kernel
        self.cupy_kernel(
            (grid_size,),
            (block_size,),
            (
                cuda_query,
                cuda_query_popcount,
                db_slice,
                popcnts_slice,
                cuda_threshold,
                cuda_num_mols,
                cuda_fp_size,
                sims,
            ),
        )
        mask = cp.where(sims > 0)[0]
        ids_slice = self.cuda_ids[slice_range]
        return cp.asnumpy(ids_slice[mask]), cp.asnumpy(sims[mask])

    def similarity(
        self, query_string: str, threshold: str, full_sanitization: bool = True
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
        np_query = self.load_query(query_string, full_sanitization=full_sanitization)
        ids, sims = self._raw_kernel_search(np_query, threshold)

        # create results numpy array
        results = np.empty(
            len(ids),
            dtype=np.dtype([("mol_id", "u4"), ("coeff", "f4")]),
        )
        results["mol_id"] = ids
        results["coeff"] = sims
        if len(results) > 0:
            results[::-1].sort(order="coeff")
        return results

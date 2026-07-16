from .io.chem import get_bounds_range, METRICS
from .base import BaseEngine
import numpy as np
import cupy as cp


class FPSim2CudaEngine(BaseEngine):
    """FPSim2 class to run fast GPU similarity searches.

    Supports every metric exposed by the CPU engine: ``tanimoto``, ``dice``,
    ``cosine`` and ``hamming``.

    Parameters
    ----------
    fp_filename : str
        Fingerprints database file path.

    fps_sort : bool
        Wheter if the FPs should be sorted after being loaded into memory or not.

    storage_backend : str
        Which storage backend to use.
    """

    # A single kernel serves every metric. The intersection popcount (``common``)
    # is metric-independent; only the final coefficient formula differs, so each
    # metric is a small ``__device__`` function selected at runtime by ``calc_type``.
    # ``calc_type`` values match ``FPSim2.io.chem.METRICS`` and the C++ backend:
    #   0 = tanimoto, 1 = dice, 2 = cosine, 3 = hamming.
    raw_kernel = r"""
        // --- per-metric coefficient functions -----------------------------------
        // Each takes the intersection popcount and both fingerprint popcounts
        // (plus the total bit length ``n_bits`` for hamming) and returns the
        // similarity coefficient.
        __device__ __forceinline__ float coeff_tanimoto(
                int common, unsigned long long int qcount,
                unsigned long long int ocount, int n_bits) {{
            float denom = (float)(qcount + ocount - common);
            return denom != 0.0f ? common / denom : 0.0f;
        }}

        __device__ __forceinline__ float coeff_dice(
                int common, unsigned long long int qcount,
                unsigned long long int ocount, int n_bits) {{
            float denom = (float)(qcount + ocount);
            return denom != 0.0f ? (2.0f * common) / denom : 0.0f;
        }}

        __device__ __forceinline__ float coeff_cosine(
                int common, unsigned long long int qcount,
                unsigned long long int ocount, int n_bits) {{
            float denom = sqrtf((float)(qcount * ocount));
            return denom != 0.0f ? common / denom : 0.0f;
        }}

        __device__ __forceinline__ float coeff_hamming(
                int common, unsigned long long int qcount,
                unsigned long long int ocount, int n_bits) {{
            // hamming distance = qcount + ocount - 2 * common
            // similarity = (N - distance) / N  (shared 0-bits count as agreement)
            int distance = (int)qcount + (int)ocount - 2 * common;
            return n_bits > 0 ? (float)(n_bits - distance) / n_bits : 0.0f;
        }}

        extern "C" __global__
        void simRAW(const unsigned long long int* __restrict__ query,
                    const unsigned long long int qcount,
                    const unsigned long long int* __restrict__ db,
                    const unsigned long long int* __restrict__ popcnts,
                    const float threshold,
                    const int calc_type,
                    const int n_bits,
                    float* __restrict__ out) {{

            // Shared block array. Only visible for threads in same block
            __shared__ int common[{block}];

            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            common[threadIdx.x] = __popcll(query[threadIdx.x] & db[tid]);

            // threads need to wait until all threads finish
            __syncthreads();

            // thread 0 in each block sums the common bits
            // and calcs the final coeff for the selected metric
            if(0 == threadIdx.x)
            {{
                int comm_sum = 0;
                for(int i=0; i<{block}; i++)
                    comm_sum += common[i];

                unsigned long long int q = qcount;
                unsigned long long int o = popcnts[blockIdx.x];

                float coeff = 0.0f;
                switch (calc_type)
                {{
                    case 0: coeff = coeff_tanimoto(comm_sum, q, o, n_bits); break;
                    case 1: coeff = coeff_dice(comm_sum, q, o, n_bits);     break;
                    case 2: coeff = coeff_cosine(comm_sum, q, o, n_bits);   break;
                    case 3: coeff = coeff_hamming(comm_sum, q, o, n_bits);  break;
                }}
                out[blockIdx.x] = coeff >= threshold ? coeff : 0.0f;
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
        # total number of fingerprint bits (needed by the hamming metric)
        self.n_bits = self.cuda_db.shape[1] * 64
        self.cupy_kernel = cp.RawKernel(
            self.raw_kernel.format(block=self.cuda_db.shape[1]),
            name="simRAW",
            options=("-std=c++14",),
        )

    def _raw_kernel_search(self, np_query, threshold, metric):
        # get the range of the molecule subset to screen
        fp_range = get_bounds_range(
            np_query, threshold, 0, 0, self.popcnt_bins, metric, N=self.n_bits
        )

        if not fp_range:
            return np.ndarray(0), np.ndarray(0)

        calc_type = METRICS[metric]
        cuda_query = cp.asarray(np_query[1:-1], dtype=cp.uint64)

        slice_range = slice(*fp_range)
        grid_size = int(fp_range[1] - fp_range[0])
        sims = cp.zeros(grid_size, dtype=cp.float32)
        block_size = self.cuda_db.shape[1]  # number of uint64 elements per fingerprint

        # run the kernel. scalars (query popcount, threshold, calc_type, n_bits)
        # are passed by value to avoid per-query device allocations and per-block
        # global-memory loads.
        self.cupy_kernel(
            (grid_size,),
            (block_size,),
            (
                cuda_query,
                np.uint64(np_query[-1]),
                self.cuda_db[slice_range],
                self.cuda_db_popcnts[slice_range],
                np.float32(threshold),
                np.int32(calc_type),
                np.int32(self.n_bits),
                sims,
            ),
        )
        # Filter and sort entirely on the GPU: sub-threshold entries are 0.0, so
        # keep only the positive ones and order them by descending coeff before
        # transferring. Only the surviving (id, coeff) pairs cross to the host,
        # avoiding a slow numpy structured-array sort afterwards.
        survivors = cp.where(sims > 0)[0]
        if survivors.size == 0:
            return (
                np.array([], dtype=np.uint32),
                np.array([], dtype=np.float32),
            )

        surv_coeffs = sims[survivors]
        surv_ids = self.cuda_ids[slice_range][survivors]
        order = cp.argsort(-surv_coeffs)  # descending
        return cp.asnumpy(surv_ids[order]), cp.asnumpy(surv_coeffs[order])

    def similarity(
        self,
        query_string: str,
        threshold: float,
        metric: str = "tanimoto",
        full_sanitization: bool = True,
    ) -> np.ndarray:
        """Runs a CUDA similarity search.

        Parameters
        ----------
        query_string : str
            SMILES, InChI or molblock.

        threshold : float
            Similarity threshold.

        metric : str
            Similarity metric to use. One of ``tanimoto``, ``dice``, ``cosine``
            or ``hamming``.

        Returns
        -------
        results : numpy array
            Similarity results.
        """
        if metric not in METRICS:
            raise ValueError(
                f"Invalid metric: {metric}. Available metrics are: {list(METRICS.keys())}"
            )
        np_query = self.load_query(query_string, full_sanitization=full_sanitization)
        ids, sims = self._raw_kernel_search(np_query, threshold, metric)

        # Results are already filtered and sorted (descending) on the GPU.
        results = np.empty(
            len(ids),
            dtype=np.dtype([("mol_id", "u4"), ("coeff", "f4")]),
        )
        results["mol_id"] = ids
        results["coeff"] = sims
        return results

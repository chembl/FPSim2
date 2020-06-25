from .io.chem import get_bounds_range
from .FPSim2lib import sort_results
from .base import BaseEngine
import numpy as np
import cupy as cp


class FPSim2CudaEngine(BaseEngine):
    """FPSim2 class to run GPU searches.

    Args:
        fp_filename: FP db file path.
        in_memory_fps: Flat to load into memory or not the fps.
        fps_sort: Flag to sort or not fps after loading into memory.
    """

    c_kernel = r"""
    extern "C" __global__
    void taniRAW(const unsigned long long int* query,
                 const unsigned long long int* qcount,
                 const unsigned long long int* db,
                 const unsigned long long int* popcnts,
                 float* threshold,
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
        fp_filename: str,
        in_memory_fps: bool = True,
        fps_sort: bool = False,
        storage_backend: str = "pytables",
    ) -> None:
        super(FPSim2CudaEngine, self).__init__(
            fp_filename=fp_filename,
            storage_backend=storage_backend,
            in_memory_fps=True,
            fps_sort=False,
        )
        self.cuda_db = cp.asarray(self.fps[:, 1:-1])
        self.cuda_ids = cp.asarray(self.fps[:, 0])
        self.cuda_popcnts = cp.asarray(self.fps[:, -1])
        self.kernel = cp.RawKernel(
            self.c_kernel.format(block=self.cuda_db.shape[1]),
            name="taniRAW",
            options=("-std=c++14", ),
        )

    def similarity(self, query_string, threshold):
        """Run a CUDA Tanimoto search

        Args:
            query_string: SMILES, InChi or molblock.
            threshold: Similarities with a coeff above the threshold will be kept.
        Returns:
            Numpy array with ids and similarities.
        """

        np_query = self.load_query(query_string)
        c_query = cp.asarray(np_query[1:-1])
        qpopcnt = cp.asarray(np_query[-1])

        # get the range of the molecule subset to screen
        fp_range = get_bounds_range(
            np_query, threshold, 0, 0, self.popcnt_bins, "tanimoto"
        )

        cuda_threshold = cp.asarray(threshold, dtype="f4")

        # get the subset of molecule ids
        subset_size = int(fp_range[1] - fp_range[0])
        ids = self.cuda_ids[slice(*fp_range)]

        # init results array
        cp_results = cp.zeros(subset_size, dtype=cp.float32)

        # run in the search, it compiles the kernel only the first time it runs
        # grid, block and arguments
        self.kernel(
            (subset_size,),
            (self.cuda_db.shape[1],),
            (
                c_query,
                qpopcnt,
                self.cuda_db[slice(*fp_range)],
                self.cuda_popcnts[slice(*fp_range)],
                cuda_threshold,
                cp_results,
            ),
        )

        # get all non 0 values and ids
        mask = cp_results.nonzero()[0]
        np_sim = cp.asnumpy(cp_results[mask])
        np_ids = cp.asnumpy(ids[mask])

        # create results numpy array
        results = np.empty(
            len(np_ids), dtype=np.dtype([("idx", "u4"), ("mol_id", "u4"), ("coeff", "f4")])
        )
        results["idx"] = np.zeros(np_ids.shape[0])
        results["mol_id"] = np_ids
        results["coeff"] = np_sim
        sort_results(results)
        return results[["mol_id", "coeff"]]

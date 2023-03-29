from .io.chem import get_bounds_range
from .FPSim2lib.utils import SortResults
from .base import BaseEngine
import numpy as np
import cupy as cp


class FPSim2CudaEngine(BaseEngine):
    """FPSim2 class to run fast CPU searches.

    Parameters
    ----------
    fp_filename : str
        Fingerprints database file path.

    fps_sort : bool
        Wheter if the FPs should be sorted after being loaded into memory or not.

    storage_backend : str
        Which storage backend to use (only pytables available).

    kernel: str
        Which CUDA kernel to use (raw or element_wise).

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

    ew_kernel = r"""
        int comm_sum = 0;
        for(int j = 1; j < in_width - 1; ++j){
            int pos = i * in_width + j;
            comm_sum += __popcll(db[pos] & query[j]);
        }
        float coeff = 0.0;
        coeff = query[in_width - 1] + db[i * in_width + in_width - 1] - comm_sum;
        if (coeff != 0.0)
            coeff = comm_sum / coeff;
        out[i] = coeff >= threshold ? coeff : 0.0;
    """

    def __init__(
        self,
        fp_filename: str = "",
        storage_backend: str = "pytables",
        kernel: str = "raw",
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
        self.kernel = kernel
        if kernel == "raw":
            # copy all the stuff to the GPU
            self.cuda_db = cp.asarray(self.fps[:, 1:-1])
            self.cuda_ids = cp.asarray(self.fps[:, 0])
            self.cuda_popcnts = cp.asarray(self.fps[:, -1])
            self.cupy_kernel = cp.RawKernel(
                self.raw_kernel.format(block=self.cuda_db.shape[1]),
                name="taniRAW",
                options=("-std=c++14",),
            )

        elif self.kernel == "element_wise":
            # copy the database to the GPU
            self.cuda_db = cp.asarray(self.fps)
            self.cupy_kernel = cp.ElementwiseKernel(
                in_params="raw T db, raw U query, raw T in_width, raw T threshold",
                out_params="raw V out",
                operation=self.ew_kernel,
                name="taniEW",
                options=("-std=c++14",),
                reduce_dims=False,
            )

        else:
            raise Exception("only supports 'raw' and 'element_wise' kernels")

    def _element_wise_search(self, query_string, threshold):

        np_query = self.load_query(query_string)

        # get the range of the molecule subset to screen
        fp_range = get_bounds_range(
            np_query, threshold, 0, 0, self.popcnt_bins, "tanimoto"
        )
        if fp_range:
            # copy query and threshold to GPU
            query = cp.asarray(np_query)

            # get the subset of molecule ids
            subset_size = int(fp_range[1] - fp_range[0])
            ids = self.cuda_db[:, 0][slice(*fp_range)]

            # init sims result array
            sims = cp.zeros(subset_size, dtype="f4")

            self.cupy_kernel(
                self.cuda_db[slice(*fp_range)],
                query,
                self.cuda_db.shape[1],
                threshold,
                sims,
                size=subset_size,
            )

            mask = sims.nonzero()[0]
            np_sim = cp.asnumpy(sims[mask])
            np_ids = cp.asnumpy(ids[mask])
        else:
            np_sim = np.ndarray(0)
            np_ids = np.ndarray(0)

        return np_ids, np_sim

    def _raw_kernel_search(self, query_string, threshold):

        np_query = self.load_query(query_string)

        # get the range of the molecule subset to screen
        fp_range = get_bounds_range(
            np_query, threshold, 0, 0, self.popcnt_bins, "tanimoto"
        )

        if fp_range:
            # copy query and threshold to GPU
            cuda_threshold = cp.asarray(threshold, dtype="f4")
            query = cp.asarray(np_query[1:-1])
            popcount = cp.asarray(np_query[-1])

            # get the subset of molecule ids
            subset_size = int(fp_range[1] - fp_range[0])
            ids = self.cuda_ids[slice(*fp_range)]

            # init sims result array
            sims = cp.zeros(subset_size, dtype=cp.float32)

            # run in the search, it compiles the kernel only the first time it runs
            # grid, block and arguments
            self.cupy_kernel(
                (subset_size,),
                (self.cuda_db.shape[1],),
                (
                    query,
                    popcount,
                    self.cuda_db[slice(*fp_range)],
                    self.cuda_popcnts[slice(*fp_range)],
                    cuda_threshold,
                    sims,
                ),
            )

            # get all non 0 values and ids
            mask = sims.nonzero()[0]
            np_sim = cp.asnumpy(sims[mask])
            np_ids = cp.asnumpy(ids[mask])
        else:
            np_sim = np.ndarray(0)
            np_ids = np.ndarray(0)

        return np_ids, np_sim

    def similarity(self, query_string: str, threshold: str) -> np.ndarray:
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
        if self.kernel == "raw":
            ids, sims = self._raw_kernel_search(query_string, threshold)
        elif self.kernel == "element_wise":
            ids, sims = self._element_wise_search(query_string, threshold)
        else:
            raise Exception("only supports 'raw' and 'element_wise' kernels")

        # create results numpy array
        results = np.empty(
            len(ids),
            dtype=np.dtype([("idx", "u4"), ("mol_id", "u4"), ("coeff", "f4")]),
        )
        results["idx"] = np.zeros(ids.shape[0])  # need to get rid of this
        results["mol_id"] = ids
        results["coeff"] = sims
        SortResults(results)

        return results[["mol_id", "coeff"]]

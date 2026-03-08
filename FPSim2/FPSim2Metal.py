"""FPSim2 Metal GPU backend for Apple Silicon (M1/M2/M3/M4).

Requires MLX: pip install mlx
"""
from .io.chem import get_bounds_range
from .FPSim2lib.utils import SortResults
from .base import BaseEngine
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class FPSim2MetalEngine(BaseEngine):
    """FPSim2 engine using Apple Metal GPU via MLX.

    Parameters
    ----------
    fp_filename : str
        Fingerprints database file path.

    storage_backend : str
        Which storage backend to use.

    Notes
    -----
    Requires Apple Silicon (M1/M2/M3/M4) and MLX installed.
    Install MLX with: pip install mlx
    """

    def __init__(
        self,
        fp_filename: str = "",
        storage_backend: str = "pytables",
        conn_url: str = "",
        table_name: str = "",
        pg_schema: str = "",
    ) -> None:
        if not HAS_MLX:
            raise ImportError(
                "MLX is required for Metal GPU support. Install with: pip install mlx"
            )

        super(FPSim2MetalEngine, self).__init__(
            fp_filename=fp_filename,
            storage_backend=storage_backend,
            in_memory_fps=True,
            fps_sort=False,
            conn_url=conn_url,
            table_name=table_name,
            pg_schema=pg_schema,
        )
        # Convert fingerprints to MLX arrays on GPU
        # Store as uint64 for bit operations
        self.metal_db = mx.array(self.fps[:, 1:-1].astype(np.uint64))
        self.metal_ids = mx.array(self.fps[:, 0].astype(np.uint32))
        self.metal_db_popcnts = mx.array(self.fps[:, -1].astype(np.uint32))
        self.fp_size = self.metal_db.shape[1]  # Number of uint64 blocks per fingerprint
        
        # Initialize Metal kernel
        self._init_metal_kernel()
    
    def _init_metal_kernel(self):
        """Initialize custom Metal kernel for Tanimoto computation."""
        # Metal kernel with popcount for Tanimoto similarity
        source = f"""
            uint tid = thread_position_in_grid.x;
            const uint fp_size_val = {self.fp_size};
            const uint num_mols_val = num_mols[0];
            
            // Each thread computes one molecule's Tanimoto coefficient
            if (tid < num_mols_val) {{
                int common_bits = 0;
                
                // Compute popcount of (query AND db_fp) for each uint64 block
                for (uint i = 0; i < fp_size_val; i++) {{
                    ulong and_val = query[i] & db[tid * fp_size_val + i];
                    common_bits += popcount(and_val);
                }}
                
                // Tanimoto = intersection / (A + B - intersection)
                int union_count = query_popcnt[0] + popcnts[tid] - common_bits;
                
                float coeff = 0.0f;
                if (union_count > 0) {{
                    coeff = (float)common_bits / (float)union_count;
                }}
                
                // Apply threshold
                out[tid] = (coeff >= threshold[0]) ? coeff : 0.0f;
            }}
        """
        
        self._tanimoto_kernel = mx.fast.metal_kernel(
            name="tanimoto_similarity",
            input_names=["query", "query_popcnt", "db", "popcnts", "threshold", "num_mols"],
            output_names=["out"],
            source=source,
        )

    def _compute_tanimoto_metal(
        self, query_fp: np.ndarray, query_popcnt: int, db_slice: mx.array, 
        popcnts_slice: mx.array, threshold: float
    ) -> tuple:
        """Compute Tanimoto similarity using custom Metal kernel.
        
        Uses Metal's native popcount for maximum GPU performance.
        """
        num_mols = db_slice.shape[0]
        
        # Flatten db_slice for kernel (contiguous memory)
        db_flat = mx.reshape(db_slice, (-1,))
        
        # Prepare inputs
        query = mx.array(query_fp.astype(np.uint64))
        query_pc = mx.array([query_popcnt], dtype=mx.uint32)
        threshold_arr = mx.array([threshold], dtype=mx.float32)
        num_mols_arr = mx.array([num_mols], dtype=mx.uint32)
        
        # Calculate grid size - one thread per molecule
        grid_size = ((num_mols + 255) // 256) * 256  # Round up to multiple of threadgroup size
        
        # Run kernel
        outputs = self._tanimoto_kernel(
            inputs=[query, query_pc, db_flat, popcnts_slice, threshold_arr, num_mols_arr],
            grid=(grid_size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(num_mols,)],
            output_dtypes=[mx.float32],
        )
        
        coeffs = outputs[0]
        mask = coeffs > 0
        
        mx.eval(coeffs, mask)
        
        return coeffs, mask

    def _metal_search(self, np_query: np.ndarray, threshold: float):
        """Perform similarity search using Metal GPU."""
        # Get the range of molecules to screen based on popcount bounds
        fp_range = get_bounds_range(
            np_query, threshold, 0, 0, self.popcnt_bins, "tanimoto"
        )

        if not fp_range:
            return np.array([]), np.array([])

        slice_start, slice_end = int(fp_range[0]), int(fp_range[1])

        # Get slices on GPU
        db_slice = self.metal_db[slice_start:slice_end]
        popcnts_slice = self.metal_db_popcnts[slice_start:slice_end]
        ids_slice = self.metal_ids[slice_start:slice_end]

        # Compute similarities using custom Metal kernel
        coeffs, mask = self._compute_tanimoto_metal(
            np_query[1:-1], 
            int(np_query[-1]), 
            db_slice, 
            popcnts_slice, 
            threshold
        )

        # Convert to numpy for filtering
        coeffs_np = np.array(coeffs)
        mask_np = np.array(mask)
        ids_np = np.array(ids_slice)
        
        # Get indices where mask is True
        indices = np.where(mask_np)[0]
        
        if len(indices) == 0:
            return np.array([], dtype=np.uint32), np.array([], dtype=np.float32)
        
        result_ids = ids_np[indices]
        result_coeffs = coeffs_np[indices]
        
        return result_ids.astype(np.uint32), result_coeffs.astype(np.float32)

    def similarity(
        self, query_string: str, threshold: float, full_sanitization: bool = True
    ) -> np.ndarray:
        """Runs a Metal GPU Tanimoto search.

        Parameters
        ----------
        query_string : str
            SMILES, InChI or molblock.

        threshold : float
            Similarity threshold.

        full_sanitization : bool
            Whether to perform full sanitization of the query molecule.

        Returns
        -------
        results : numpy array
            Similarity results as structured array with 'mol_id' and 'coeff' fields.
        """
        np_query = self.load_query(query_string, full_sanitization=full_sanitization)
        ids, sims = self._metal_search(np_query, threshold)

        # Create results numpy array
        results = np.empty(
            len(ids),
            dtype=np.dtype([("mol_id", "u4"), ("coeff", "f4")]),
        )
        if len(ids) > 0:
            results["mol_id"] = ids
            results["coeff"] = sims
            SortResults(results)
        
        return results

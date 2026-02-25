"""Tests for the Parquet storage backend."""

import pytest
import numpy as np
import os
import tempfile

# Skip all tests if pyarrow is not installed
pytest.importorskip("pyarrow")

from FPSim2 import FPSim2Engine
from FPSim2.io.backends.parquet import (
    ParquetStorageBackend,
    create_parquet_file,
    h5_to_parquet,
)


@pytest.fixture
def smi_file():
    return "tests/data/10mols.smi"


@pytest.fixture
def h5_file():
    # Use a pre-existing static HDF5 test file to avoid inter-test dependencies
    return "tests/data/test.h5"


@pytest.fixture
def parquet_file(smi_file):
    """Create a temporary Parquet file from test molecules."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        parquet_path = f.name
    
    create_parquet_file(
        mols_source=smi_file,
        filename=parquet_path,
        mol_format="smi",
        fp_type="Morgan",
        fp_params={"radius": 2, "fpSize": 2048},
    )
    
    yield parquet_path
    
    # Cleanup
    if os.path.exists(parquet_path):
        os.remove(parquet_path)


@pytest.fixture
def parquet_from_h5(h5_file):
    """Create a temporary Parquet file converted from HDF5."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        parquet_path = f.name
    
    h5_to_parquet(h5_file, parquet_path)
    
    yield parquet_path
    
    if os.path.exists(parquet_path):
        os.remove(parquet_path)


class TestParquetBackend:
    """Test ParquetStorageBackend functionality."""

    def test_create_parquet_file(self, parquet_file):
        """Test that Parquet file is created correctly."""
        import pyarrow.parquet as pq
        
        pf = pq.ParquetFile(parquet_file)
        assert pf.metadata.num_rows == 10
        
        # Check schema has expected columns
        schema = pf.schema_arrow
        assert "mol_id" in schema.names
        assert "popcnt" in schema.names
        assert "f1" in schema.names  # At least one fingerprint column

    def test_load_parquet_backend(self, parquet_file):
        """Test loading Parquet file into ParquetStorageBackend."""
        backend = ParquetStorageBackend(parquet_file)
        
        assert backend.fps is not None
        assert backend.fps.shape[0] == 10
        assert backend.popcnt_bins is not None

    def test_fpsim2_engine_parquet(self, parquet_file):
        """Test FPSim2Engine with Parquet backend."""
        fpe = FPSim2Engine(parquet_file, storage_backend="parquet")
        
        assert fpe.fps.shape[0] == 10
        assert fpe.fp_type == "Morgan"
        assert fpe.fp_params["radius"] == 2

    def test_similarity_search_parquet(self, parquet_file):
        """Test similarity search with Parquet backend."""
        fpe = FPSim2Engine(parquet_file, storage_backend="parquet")
        
        # Use first molecule as query
        results = fpe.similarity("Cc1ccc(-n2ncc(=O)[nH]c2=O)cc1", 0.5)
        
        assert len(results) > 0
        assert all(r[1] >= 0.5 for r in results)  # All results above threshold

    def test_h5_to_parquet_conversion(self, h5_file, parquet_from_h5):
        """Test HDF5 to Parquet conversion produces equivalent results."""
        # Load both backends
        fpe_h5 = FPSim2Engine(h5_file, storage_backend="pytables")
        fpe_pq = FPSim2Engine(parquet_from_h5, storage_backend="parquet")
        
        # Compare fingerprint arrays
        np.testing.assert_array_equal(fpe_h5.fps, fpe_pq.fps)
        
        # Compare metadata
        assert fpe_h5.fp_type == fpe_pq.fp_type
        assert fpe_h5.fp_params == fpe_pq.fp_params

    def test_search_results_match_h5(self, h5_file, parquet_from_h5):
        """Test that Parquet backend produces same search results as HDF5."""
        fpe_h5 = FPSim2Engine(h5_file, storage_backend="pytables")
        fpe_pq = FPSim2Engine(parquet_from_h5, storage_backend="parquet")
        
        query = "Cc1ccc(-n2ncc(=O)[nH]c2=O)cc1"
        threshold = 0.5
        
        results_h5 = fpe_h5.similarity(query, threshold)
        results_pq = fpe_pq.similarity(query, threshold)
        
        # Same number of results
        assert len(results_h5) == len(results_pq)
        
        # Same mol_ids (may be in different order due to floating point)
        h5_ids = set(r[0] for r in results_h5)
        pq_ids = set(r[0] for r in results_pq)
        assert h5_ids == pq_ids

    def test_metadata_from_parquet(self, parquet_file):
        """Test that metadata is correctly read from Parquet file."""
        backend = ParquetStorageBackend(parquet_file)
        
        assert backend.fp_type == "Morgan"
        assert backend.fp_params["radius"] == 2
        assert backend.fp_params["fpSize"] == 2048
        assert backend.rdkit_ver is not None

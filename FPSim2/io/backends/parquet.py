"""Parquet storage backend for FPSim2."""

from typing import Dict, Iterable as IterableType, Tuple, Union
from .base import BaseStorageBackend
import numpy as np
import rdkit
import math
import json
import time
from importlib.metadata import version

__version__ = version("FPSim2")

try:
    import pyarrow.parquet as pq
    import pyarrow as pa

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def create_parquet_file(
    mols_source: Union[str, IterableType],
    filename: str,
    mol_format: str,
    fp_type: str,
    fp_params: dict = {},
    mol_id_prop: str = "mol_id",
    full_sanitization: bool = True,
    compression: str = "zstd",
    compression_level: int = 3,
    row_group_size: int = 100000,
) -> None:
    """Create a Parquet fingerprint database file."""
    if not HAS_PYARROW:
        raise ImportError("PyArrow is required. Install with: pip install pyarrow")

    from ..chem import (
        build_fp,
        get_mol_supplier,
        get_fp_length,
        FP_FUNC_DEFAULTS,
        RDKIT_PARSE_FUNCS,
    )

    is_valid_file = isinstance(mols_source, str) and (
        mols_source.endswith((".smi", ".sdf", ".sdf.gz"))
    )
    if not (is_valid_file or mol_format in RDKIT_PARSE_FUNCS):
        raise ValueError(f"Unsupported mol_format: {mol_format}")

    if fp_type not in FP_FUNC_DEFAULTS:
        raise ValueError(f"Unsupported fp_type: {fp_type}")

    if not fp_params:
        fp_params = FP_FUNC_DEFAULTS[fp_type]
    else:
        if "fpSize" not in fp_params:
            if "fpSize" in FP_FUNC_DEFAULTS[fp_type]:
                fp_params["fpSize"] = FP_FUNC_DEFAULTS[fp_type]["fpSize"]

    supplier = get_mol_supplier(mols_source)
    fp_length = get_fp_length(fp_type, fp_params)
    n_fp_cols = math.ceil(fp_length / 64)

    # Build schema with metadata
    fields = [("mol_id", pa.int64())]
    fields += [(f"f{i + 1}", pa.uint64()) for i in range(n_fp_cols)]
    fields += [("popcnt", pa.int64())]
    schema = pa.schema(fields)

    metadata = {
        b"fp_type": fp_type.encode(),
        b"fp_params": json.dumps(fp_params).encode(),
        b"rdkit_version": rdkit.__version__.encode(),
        b"fpsim2_version": __version__.encode(),
    }
    schema = schema.with_metadata(metadata)

    iterable = supplier(
        mols_source,
        full_sanitization,
        mol_format=mol_format,
        mol_id_prop=mol_id_prop,
    )

    # Write in batches to avoid loading all into memory
    fps_batch = []
    rows_written = 0

    with pq.ParquetWriter(
        filename,
        schema,
        compression=compression,
        compression_level=compression_level if compression else None,
    ) as writer:
        for mol_id, rdmol in iterable:
            fp = build_fp(rdmol, fp_type, fp_params, mol_id)
            fps_batch.append(fp)

            if len(fps_batch) >= row_group_size:
                _write_batch(writer, fps_batch, n_fp_cols, schema)
                rows_written += len(fps_batch)
                fps_batch = []

        # Write remaining batch
        if fps_batch:
            _write_batch(writer, fps_batch, n_fp_cols, schema)
            rows_written += len(fps_batch)

    if rows_written == 0:
        raise ValueError("No valid molecules found in source")


def _write_batch(writer, fps_batch, n_fp_cols, schema):
    """Write a batch of fingerprints to Parquet."""
    fps_array = np.array(fps_batch, dtype=np.uint64)
    arrays = {"mol_id": pa.array(fps_array[:, 0].astype(np.int64))}
    for i in range(n_fp_cols):
        arrays[f"f{i + 1}"] = pa.array(fps_array[:, i + 1])
    arrays["popcnt"] = pa.array(fps_array[:, -1].astype(np.int64))
    table = pa.table(arrays, schema=schema)
    writer.write_table(table)


def h5_to_parquet(
    h5_filename: str,
    parquet_filename: str,
    compression: str = "zstd",
    compression_level: int = 3,
    row_group_size: int = 100000,
) -> None:
    """Convert an existing HDF5/PyTables FPSim2 database to Parquet format."""
    if not HAS_PYARROW:
        raise ImportError("PyArrow is required. Install with: pip install pyarrow")

    import tables as tb
    from .pytables import get_fp_length

    with tb.open_file(h5_filename, mode="r") as fp_file:
        fp_type = fp_file.root.config[0]
        fp_params = fp_file.root.config[1]
        rdkit_ver = fp_file.root.config[2]
        fpsim2_ver = fp_file.root.config[3]

        fp_length = get_fp_length(fp_type, fp_params)
        n_fp_cols = math.ceil(fp_length / 64)
        n_rows = fp_file.root.fps.nrows
        num_fields = len(fp_file.root.fps.dtype)

        # Build schema with metadata
        fields = [("mol_id", pa.int64())]
        fields += [(f"f{i + 1}", pa.uint64()) for i in range(n_fp_cols)]
        fields += [("popcnt", pa.int64())]
        schema = pa.schema(fields)

        metadata = {
            b"fp_type": fp_type.encode(),
            b"fp_params": json.dumps(fp_params).encode(),
            b"rdkit_version": rdkit_ver.encode(),
            b"fpsim2_version": fpsim2_ver.encode(),
        }
        schema = schema.with_metadata(metadata)

        # Write in chunks to avoid loading all into memory
        with pq.ParquetWriter(
            parquet_filename,
            schema,
            compression=compression,
            compression_level=compression_level if compression else None,
        ) as writer:
            for start in range(0, n_rows, row_group_size):
                end = min(start + row_group_size, n_rows)
                chunk = fp_file.root.fps[start:end]
                fps_array = chunk.view("<u8").reshape(-1, num_fields)

                arrays = {"mol_id": pa.array(fps_array[:, 0].astype(np.int64))}
                for i in range(n_fp_cols):
                    arrays[f"f{i + 1}"] = pa.array(fps_array[:, i + 1])
                arrays["popcnt"] = pa.array(fps_array[:, -1].astype(np.int64))

                table = pa.table(arrays, schema=schema)
                writer.write_table(table)


class ParquetStorageBackend(BaseStorageBackend):
    """Parquet storage backend for fingerprint data."""

    def __init__(self, fp_filename: str, fps_sort: bool = True) -> None:
        if not HAS_PYARROW:
            raise ImportError("PyArrow is required. Install with: pip install pyarrow")

        super(ParquetStorageBackend, self).__init__()
        self.name = "parquet"
        self.fp_filename = fp_filename
        self.fp_type, self.fp_params, self.rdkit_ver, self.fpsim2_ver = (
            self.read_parameters()
        )
        self.load_fps()
        self.load_popcnt_bins()

        if self.rdkit_ver and self.rdkit_ver != rdkit.__version__:
            print(
                f"Warning: Database was created with RDKit version {self.rdkit_ver} "
                f"but installed version is {rdkit.__version__}. "
                "Please ensure there were no relevant changes in RDKit regarding "
                "fingerprint generation between these versions."
            )

    def read_parameters(self) -> Tuple[str, Dict, str, str]:
        """Read fingerprint parameters from file metadata."""
        pf = pq.ParquetFile(self.fp_filename)
        metadata = pf.schema_arrow.metadata or {}

        def get_meta(key: str) -> str:
            val = metadata.get(key.encode(), b"")
            return val.decode() if val else None

        fp_type = get_meta("fp_type")
        fp_params_str = get_meta("fp_params")
        fp_params = json.loads(fp_params_str) if fp_params_str else {}
        rdkit_ver = get_meta("rdkit_version")
        fpsim2_ver = get_meta("fpsim2_version")

        return fp_type, fp_params, rdkit_ver, fpsim2_ver

    def load_fps(self) -> None:
        """Load fingerprints from Parquet file into memory."""
        # Time loading the file
        load_start = time.time()
        pf = pq.ParquetFile(self.fp_filename)
        schema = pf.schema_arrow
        fp_cols = sorted(
            [f.name for f in schema if f.name.startswith("f") and f.name[1:].isdigit()],
            key=lambda x: int(x[1:]),
        )
        columns = ["mol_id"] + fp_cols + ["popcnt"]

        # Build structured dtype for in-place sort
        dtype = [("fp_id", "<i8")]
        dtype += [(f"f{i+1}", "<u8") for i in range(len(fp_cols))]
        dtype += [("popcnt", "<i8")]

        n_rows = pf.metadata.num_rows
        load_end = time.time()
        print(f"Time to load Parquet file: {load_end - load_start:.4f} seconds")

        # Time converting to numpy
        convert_start = time.time()
        # Create structured array and fill column by column (avoids full table copy)
        fps = np.empty(n_rows, dtype=dtype)
        fps["fp_id"] = pf.read(columns=["mol_id"]).column(0).to_numpy()
        for col in fp_cols:
            fps[col] = pf.read(columns=[col]).column(0).to_numpy()
        fps["popcnt"] = pf.read(columns=["popcnt"]).column(0).to_numpy()
        convert_end = time.time()
        print(f"Time to convert to numpy: {convert_end - convert_start:.4f} seconds")

        # Time sorting
        sort_start = time.time()
        # In-place sort (no copy)
        fps.sort(order="popcnt")
        sort_end = time.time()
        print(f"Time to sort: {sort_end - sort_start:.4f} seconds")

        # View as 2D uint64 (no copy)
        self.fps = fps.view("<u8").reshape(-1, len(columns))

    def load_popcnt_bins(self) -> None:
        self.popcnt_bins = self.calc_popcnt_bins(self.fps)
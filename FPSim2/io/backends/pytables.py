from typing import Any, Iterable as IterableType, Dict, List, Tuple, Union
from .base import BaseStorageBackend
from ..chem import (
    build_fp,
    get_mol_supplier,
    get_fp_length,
    it_mol_supplier,
    FP_FUNC_DEFAULTS,
    RDKIT_PARSE_FUNCS,
)
import tables as tb
import numpy as np
import rdkit
import math
import os
from importlib.metadata import version

__version__ = version("FPSim2")

BATCH_WRITE_SIZE = 32000


def create_schema(fp_length: int) -> Any:
    class Particle(tb.IsDescription):
        pass

    columns = {}
    pos = 1
    columns["fp_id"] = tb.Int64Col(pos=pos)
    for i in range(1, math.ceil(fp_length / 64) + 1):
        pos += 1
        columns["f" + str(i)] = tb.UInt64Col(pos=pos)
    columns["popcnt"] = tb.Int64Col(pos=pos + 1)
    Particle.columns = columns
    return Particle


def create_db_file(
    mols_source: Union[str, IterableType],
    filename: str,
    mol_format: str,
    fp_type: str,
    fp_params: dict = {},
    mol_id_prop: str = "mol_id",
    sort_by_popcnt: bool = True,
) -> None:
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
    filters = tb.Filters(complib="blosc2", complevel=9, fletcher32=False)

    with tb.open_file(filename, mode="w") as fp_file:
        particle = create_schema(fp_length)
        fps_table = fp_file.create_table(
            fp_file.root, "fps", particle, "Table storing fps", filters=filters
        )

        param_table = fp_file.create_vlarray(
            fp_file.root, "config", atom=tb.ObjectAtom()
        )
        param_table.append(fp_type)
        param_table.append(fp_params)
        param_table.append(rdkit.__version__)
        param_table.append(__version__)

        fps = []
        iterable = supplier(mols_source, mol_format=mol_format, mol_id_prop=mol_id_prop)
        for mol_id, rdmol in iterable:
            fp = build_fp(rdmol, fp_type, fp_params, mol_id)
            fps.append(fp)
            if len(fps) == BATCH_WRITE_SIZE:
                fps_table.append(fps)
                fps = []
        if fps:
            fps_table.append(fps)

        fps_table.cols.popcnt.create_index(kind="full")

    if sort_by_popcnt:
        sort_db_file(filename)


def calc_popcnt_bins_pytables(fps: Any, fp_length: int) -> list:
    popcnt_bins = []
    for i in range(0, fp_length + 1):
        idx_gen = (row.nrow for row in fps.where("popcnt == {}".format(str(i))))
        try:
            first_id = next(idx_gen)
        except StopIteration:
            continue
        j = first_id
        for j in idx_gen:
            pass
        cnt_idxs = (first_id, j + 1)
        popcnt_bins.append((i, cnt_idxs))
    return popcnt_bins


def sort_db_file(filename: str) -> None:
    """Sorts the FPs db file."""
    # rename not sorted filename
    tmp_filename = filename + "_tmp"
    os.rename(filename, tmp_filename)
    filters = tb.Filters(complib="blosc", complevel=5)

    # copy sorted fps and config to a new file
    with tb.open_file(tmp_filename, mode="r") as fp_file:
        with tb.open_file(filename, mode="w") as sorted_fp_file:
            fp_type = fp_file.root.config[0]
            fp_params = fp_file.root.config[1]
            fp_length = get_fp_length(fp_type, fp_params)

            # create a sorted copy of the fps table
            dst_fps = fp_file.root.fps.copy(
                sorted_fp_file.root,
                "fps",
                filters=filters,
                copyuserattrs=True,
                overwrite=True,
                stats={
                    "groups": 0,
                    "leaves": 0,
                    "links": 0,
                    "bytes": 0,
                    "hardlinks": 0,
                },
                start=None,
                stop=None,
                step=None,
                chunkshape="auto",
                sortby="popcnt",
                check_CSI=True,
                propindexes=True,
            )

            # set config table; used fp function, parameters and rdkit version
            param_table = sorted_fp_file.create_vlarray(
                sorted_fp_file.root, "config", atom=tb.ObjectAtom()
            )
            param_table.append(fp_type)
            param_table.append(fp_params)
            param_table.append(rdkit.__version__)
            param_table.append(__version__)

            # update count ranges
            popcnt_bins = calc_popcnt_bins_pytables(dst_fps, fp_length)
            param_table.append(popcnt_bins)

    # remove unsorted file
    os.remove(tmp_filename)


def merge_db_files(
    input_files: List[str], output_file: str, sort_by_popcnt: bool = True
) -> None:
    """Merges multiple FPs db files into a new one.

    Parameters
    ----------
    input_files : List[str]
        List of paths to input files
    output_file : str
        Path to output merged file
    sort_by_popcnt : bool, optional
        Whether to sort the output file by population count, by default True
    """
    if len(input_files) < 2:
        raise ValueError("At least two input files are required for merging")

    # Check that all files have same fingerprint type, parameters and RDKit version
    reference_configs = None
    for file in input_files:
        with tb.open_file(file, mode="r") as f:
            current_configs = (
                f.root.config[0],
                f.root.config[1],
                f.root.config[2],
                f.root.config[3],
            )
            if reference_configs is None:
                reference_configs = current_configs
            elif current_configs != reference_configs:
                raise ValueError(
                    f"File {file} has different fingerprint types, parameters or RDKit versions"
                )

    # Create new file with same parameters
    filters = tb.Filters(complib="blosc2", complevel=9, fletcher32=False)
    fp_type, fp_params, original_rdkit_ver, original_fpsim2_ver = reference_configs
    fp_length = get_fp_length(fp_type, fp_params)

    with tb.open_file(output_file, mode="w") as out_file:
        particle = create_schema(fp_length)
        fps_table = out_file.create_table(
            out_file.root, "fps", particle, "Table storing fps", filters=filters
        )

        # Copy config with original RDKit version
        param_table = out_file.create_vlarray(
            out_file.root, "config", atom=tb.ObjectAtom()
        )
        param_table.append(fp_type)
        param_table.append(fp_params)
        param_table.append(original_rdkit_ver)
        param_table.append(original_fpsim2_ver)

        # Copy data from all input files
        for file in input_files:
            with tb.open_file(file, mode="r") as in_file:
                fps_table.append(in_file.root.fps[:])

        fps_table.cols.popcnt.create_index(kind="full")

    if sort_by_popcnt:
        sort_db_file(output_file)


def migrate_db_file(original_file: str, new_file: str) -> None:
    """Migrates old database file. Use at your own risk.

    Parameters
    ----------
    original_file : str
        Path to original database file
    new_file : str
        Path to new database file
    """
    with tb.open_file(original_file, mode="r") as orig_file:
        fp_type = orig_file.root.config[0]
        fp_params = orig_file.root.config[1].copy()
        if "nBits" in fp_params:
            fp_params["fpSize"] = fp_params.pop("nBits")
        rdkit_ver = orig_file.root.config[2]

        filters = tb.Filters(complib="blosc2", complevel=9, fletcher32=False)
        fp_length = get_fp_length(fp_type, fp_params)

        with tb.open_file(new_file, mode="w") as new_fp_file:
            particle = create_schema(fp_length)
            fps_table = new_fp_file.create_table(
                new_fp_file.root, "fps", particle, "Table storing fps", filters=filters
            )

            param_table = new_fp_file.create_vlarray(
                new_fp_file.root, "config", atom=tb.ObjectAtom()
            )
            param_table.append(fp_type)
            param_table.append(fp_params)
            param_table.append(rdkit_ver)
            param_table.append(__version__)

            fps_table.append(orig_file.root.fps[:])
            fps_table.cols.popcnt.create_index(kind="full")

            popcnt_bins = calc_popcnt_bins_pytables(fps_table, fp_length)
            param_table.append(popcnt_bins)


class PyTablesStorageBackend(BaseStorageBackend):
    def __init__(
        self, fp_filename: str, in_memory_fps: bool = True, fps_sort: bool = False
    ) -> None:
        super(PyTablesStorageBackend, self).__init__()
        self.name = "pytables"
        self.fp_filename = fp_filename
        self.fp_type, self.fp_params, self.rdkit_ver, self.fpsim2_ver = (
            self.read_parameters()
        )
        if in_memory_fps:
            self.load_fps(fps_sort)
        self.load_popcnt_bins(fps_sort)
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            self.chunk_size = fp_file.root.fps.chunkshape[0] * 120
        if self.rdkit_ver != rdkit.__version__:
            print(
                f"Warning: Database was created with RDKit version {self.rdkit_ver} but installed version is {rdkit.__version__}. "
                "Please ensure there were no relevant changes in RDKit regarding fingerprint generation between these versions."
            )

    def read_parameters(self) -> Tuple[str, Dict[str, Dict[str, dict]], str]:
        """Reads fingerprint parameters"""
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            fp_type = fp_file.root.config[0]
            fp_params = fp_file.root.config[1]
            rdkit_ver = fp_file.root.config[2]
            fpsim2_ver = fp_file.root.config[3]
            try:
                fp_file.root.config[4]
            except IndexError:
                raise ValueError(
                    "Database file was generated with FPSim2 version prior to 0.6.0 and needs to be re-generated"
                )
        return fp_type, fp_params, rdkit_ver, fpsim2_ver

    def get_fps_chunk(self, chunk_range: Tuple[int, int]) -> np.asarray:
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            fps = fp_file.root.fps[slice(*chunk_range)]
        return fps

    def load_popcnt_bins(self, fps_sort) -> None:
        if fps_sort:
            popcnt_bins = self.calc_popcnt_bins(self.fps)
        else:
            with tb.open_file(self.fp_filename, mode="r") as fp_file:
                popcnt_bins = fp_file.root.config[4]
        self.popcnt_bins = popcnt_bins

    def load_fps(self, fps_sort) -> None:
        """Loads FP db file into memory.

        Parameters
        ----------
        fps_sort: bool
            Whether if the FPs should be sorted or not.

        Returns
        -------
        fps: numpy array
            Numpy array with the fingerprints.
        """
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            fps = fp_file.root.fps[:]
            # files should be sorted but if the file is updated without sorting it
            # can be also in memory sorted
            if fps_sort:
                fps.sort(order="popcnt")
        num_fields = len(fps[0])
        fps = fps.view("<u8")
        fps = fps.reshape(int(fps.size / num_fields), num_fields)
        self.fps = fps

    def delete_fps(self, ids_list: List[int]) -> None:
        """Delete FPs given a list of ids.

        Parameters
        ----------
        ids_list : list
            ids to delete.

        Returns
        -------
        None
        """
        with tb.open_file(self.fp_filename, mode="a") as fp_file:
            fps_table = fp_file.root.fps
            for fp_id in ids_list:
                to_delete = [
                    row.nrow
                    for row in fps_table.where("fp_id == {}".format(str(fp_id)))
                ]
                fps_table.remove_row(to_delete[0])

    def append_fps(self, mols_source: Union[str, IterableType], mol_format) -> None:
        """Appends FPs to the file.

        Parameters
        ----------
        mols_source : str or iterable
            .smi or .sdf filename or iterable.

        Returns
        -------
        None
        """
        fp_type, fp_params, _, _ = self.read_parameters()
        with tb.open_file(self.fp_filename, mode="a") as fp_file:
            fps_table = fp_file.root.fps
            fps = []
            for mol_id, rdmol in it_mol_supplier(mols_source, mol_format=mol_format):
                if not rdmol:
                    continue
                fp = build_fp(rdmol, fp_type, fp_params, mol_id)
                fps.append(fp)
                if len(fps) == BATCH_WRITE_SIZE:
                    fps_table.append(fps)
                    fps = []
            # append last batch < 32k
            if fps:
                fps_table.append(fps)

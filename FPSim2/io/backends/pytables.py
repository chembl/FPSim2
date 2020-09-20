from typing import Any, Iterable as IterableType, Dict, List, Tuple, Union
from .base import BaseStorageBackend
from ..chem import (
    build_fp,
    get_mol_supplier,
    get_fp_length,
    FP_FUNC_DEFAULTS,
)
import tables as tb
import numpy as np
import rdkit
import math
import os

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
    fp_type: str,
    fp_params: dict = {},
    mol_id_prop: str = "mol_id",
    gen_ids: bool = False,
    sort_by_popcnt: bool = True,
) -> None:
    """Creates FPSim2 FPs db file from .smi, .sdf files or from an iterable.

    Parameters
    ----------
    mols_source : str
        .smi/.sdf filename or iterable.

    filename: float
        Fingerprint database filename.

    fp_type : str
        Fingerprint type used to create the fingerprints.

    fp_params : dict
        Parameters used to create the fingerprints.

    mol_id_prop : str
        Name of the .sdf property to read the molecule id.

    gen_ids : bool
        Autogenerate FP ids.

    sort_by_popcnt: bool
        Whether if the FPs should be sorted or not.

    Returns
    -------
    None
    """
    # if params dict is empty use defaults
    if not fp_params:
        fp_params = FP_FUNC_DEFAULTS[fp_type]
    supplier = get_mol_supplier(mols_source)
    fp_length = get_fp_length(fp_type, fp_params)
    # set compression
    filters = tb.Filters(complib="blosc", complevel=5)

    # set the output file and fps table
    with tb.open_file(filename, mode="w") as fp_file:
        particle = create_schema(fp_length)
        fps_table = fp_file.create_table(
            fp_file.root, "fps", particle, "Table storing fps", filters=filters
        )

        # set config table; used fp function, parameters and rdkit version
        param_table = fp_file.create_vlarray(
            fp_file.root, "config", atom=tb.ObjectAtom()
        )
        param_table.append(fp_type)
        param_table.append(fp_params)
        param_table.append(rdkit.__version__)

        fps = []
        for mol_id, rdmol in supplier(mols_source, gen_ids, mol_id_prop=mol_id_prop):
            fp = build_fp(rdmol, fp_type, fp_params, mol_id)
            fps.append(fp)
            if len(fps) == BATCH_WRITE_SIZE:
                fps_table.append(fps)
                fps = []
        # append last batch < 32k
        if fps:
            fps_table.append(fps)

        # create index so table can be sorted
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

            # update count ranges
            popcnt_bins = calc_popcnt_bins_pytables(dst_fps, fp_length)
            param_table.append(popcnt_bins)

    # remove unsorted file
    os.remove(tmp_filename)


class PyTablesStorageBackend(BaseStorageBackend):
    def __init__(self, fp_filename: str, in_memory_fps: bool = True, fps_sort: bool = False) -> None:
        super(PyTablesStorageBackend, self).__init__(fp_filename)
        self.name = "pytables"
        self.fp_type, self.fp_params, self.rdkit_ver = self.read_parameters()
        if in_memory_fps:
            self.load_fps(in_memory_fps, fps_sort)
        self.load_popcnt_bins(fps_sort)
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            self.chunk_size = fp_file.root.fps.chunkshape[0] * 120

    def read_parameters(self) -> Tuple[str, Dict[str, Dict[str, dict]], str]:
        """Reads fingerprint parameters"""
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            fp_type = fp_file.root.config[0]
            fp_params = fp_file.root.config[1]
            rdkit_ver = fp_file.root.config[2]
        return fp_type, fp_params, rdkit_ver

    def get_fps_chunk(self, chunk_range: Tuple[int, int]) -> np.asarray:
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            fps = fp_file.root.fps[slice(*chunk_range)]
        return fps

    def load_popcnt_bins(self, fps_sort) -> None:
        if fps_sort:
            popcnt_bins = self.calc_popcnt_bins(self.fps)
        else:
            with tb.open_file(self.fp_filename, mode="r") as fp_file:
                popcnt_bins = fp_file.root.config[3]
        self.popcnt_bins = popcnt_bins

    def load_fps(self, in_memory_fps, fps_sort) -> None:
        """Loads FP db file into memory.

        Parameters
        ----------
        in_memory_fps : bool
            Whether if the FPs should be loaded into memory or not.

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

    def append_fps(self, mols_source: Union[str, IterableType], mol_id_prop: str = "mol_id") -> None:
        """Appends FPs to the file.

        Parameters
        ----------
        mols_source : str or iterable
            .smi or .sdf filename or iterable.

        Returns
        -------
        None
        """
        supplier = get_mol_supplier(mols_source)
        fp_type, fp_params, _ = self.read_parameters()
        with tb.open_file(self.fp_filename, mode="a") as fp_file:
            fps_table = fp_file.root.fps
            fps = []
            for mol_id, rdmol in supplier(mols_source, False, mol_id_prop=mol_id_prop):
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

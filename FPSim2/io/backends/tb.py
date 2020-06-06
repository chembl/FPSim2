from .base import BaseStorageBackend
from FPSim2.FPSim2lib import py_popcount
from ..chem import (
    get_mol_suplier,
    get_fp_length,
    rdmol_to_efp,
    calc_count_ranges,
    FP_FUNC_DEFAULTS,
)
from collections import namedtuple
import tables as tb
import numpy as np
import rdkit
import math
import os

BATCH_WRITE_SIZE = 10000


def create_db_file(
    mols_source,
    filename,
    fp_func,
    fp_func_params={},
    mol_id_prop="mol_id",
    gen_ids=False,
    sort_by_popcnt=True,
):
    """Creates FPSim2 FPs db file from .smi, .sdf files or from any iterable.

    Args:
        mols_source: .smi, .sdf filename or iterable.
        filename: FPs output filename.
        fp_func: Name of fingerprint function to use to generate fingerprints.
        fp_func_params: Parameters for the fingerprint function.
        mol_id_prop: Name of the .sdf property to read the molecule id.
        gen_ids: Flag to auto-generate ids for the molecules.
        sort_by_popcnt: Flag to sort or not fps by popcnt.
    Returns:
        None.
    """
    # if params dict is empty use defaults
    if not fp_func_params:
        fp_func_params = FP_FUNC_DEFAULTS[fp_func]
    supplier = get_mol_suplier(mols_source)
    fp_length = get_fp_length(fp_func, fp_func_params)
    # set compression
    filters = tb.Filters(complib="blosc", complevel=5)

    # set the output file and fps table
    with tb.open_file(filename, mode="w") as fp_file:

        class Particle(tb.IsDescription):
            pass

        # hacky...
        columns = {}
        pos = 1
        columns["fp_id"] = tb.Int64Col(pos=pos)
        for i in range(1, math.ceil(fp_length / 64) + 1):
            pos += 1
            columns["f" + str(i)] = tb.UInt64Col(pos=pos)
        columns["popcnt"] = tb.Int64Col(pos=pos + 1)
        Particle.columns = columns

        fps_table = fp_file.create_table(
            fp_file.root, "fps", Particle, "Table storing fps", filters=filters
        )

        # set config table; used fp function, parameters and rdkit version
        param_table = fp_file.create_vlarray(
            fp_file.root, "config", atom=tb.ObjectAtom()
        )
        param_table.append(fp_func)
        param_table.append(fp_func_params)
        param_table.append(rdkit.__version__)

        fps = []
        for mol_id, rdmol in supplier(mols_source, gen_ids, mol_id_prop=mol_id_prop):
            efp = rdmol_to_efp(rdmol, fp_func, fp_func_params)
            popcnt = py_popcount(np.array(efp, dtype=np.uint64))
            efp.insert(0, mol_id)
            efp.append(popcnt)
            fps.append(tuple(efp))
            if len(fps) == BATCH_WRITE_SIZE:
                fps_table.append(fps)
                fps = []
        # append last batch < 10k
        fps_table.append(fps)

        # create index so table can be sorted
        fps_table.cols.popcnt.create_index(kind="full")

    if sort_by_popcnt:
        sort_db_file(filename)


def sort_db_file(filename):
    """Sorts the FPs db file."""
    # rename not sorted filename
    tmp_filename = filename + "_tmp"
    os.rename(filename, tmp_filename)
    filters = tb.Filters(complib="blosc", complevel=5)

    # copy sorted fps and config to a new file
    with tb.open_file(tmp_filename, mode="r") as fp_file:
        with tb.open_file(filename, mode="w") as sorted_fp_file:
            fp_func = fp_file.root.config[0]
            fp_func_params = fp_file.root.config[1]
            fp_length = get_fp_length(fp_func, fp_func_params)

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
                chunkshape="keep",
                sortby="popcnt",
                check_CSI=True,
                propindexes=True,
            )

            # set config table; used fp function, parameters and rdkit version
            param_table = sorted_fp_file.create_vlarray(
                sorted_fp_file.root, "config", atom=tb.ObjectAtom()
            )
            param_table.append(fp_func)
            param_table.append(fp_func_params)
            param_table.append(rdkit.__version__)

            # update count ranges
            count_ranges = calc_count_ranges(dst_fps, fp_length)
            param_table.append(count_ranges)

    # remove not sorted file
    os.remove(tmp_filename)


class PyTablesStorageBackend(BaseStorageBackend):
    def __init__(self, fp_filename, in_memory_fps=True, fps_sort=False):
        super(PyTablesStorageBackend, self).__init__(
            fp_filename, in_memory_fps, fps_sort
        )

    def read_parameters(self):
        """Reads fingerprint parameters"""
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            fp_type = fp_file.root.config[0]
            fp_params = fp_file.root.config[1]
            rdkit_ver = fp_file.root.config[2]
        return fp_type, fp_params, rdkit_ver

    def get_count_ranges(self):
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            count_ranges = fp_file.root.config[3]
        return count_ranges

    def get_fps_chunk(self, chunk_range):
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            fps = fp_file.root.fps[slice(*chunk_range)]
        return fps

    def load_fps(self):
        """Loads FP db file into memory.

        Args:
        Returns:
            namedtuple with fps and count ranges.
        """
        fp_type, fp_params, rdkit_ver = self.read_parameters()
        with tb.open_file(self.fp_filename, mode="r") as fp_file:
            fps = fp_file.root.fps[:]
            # files should be sorted but if the file is updated without sorting it
            # can be also in memory sorted
            if self.fps_sort:
                fps.sort(order="popcnt")
                fp_length = get_fp_length(fp_type, fp_params)
                count_ranges = calc_count_ranges(fps, fp_length, self.in_memory_fps)
            else:
                count_ranges = fp_file.root.config[3]
        num_fields = len(fps[0])
        fps = fps.view("<u8")
        fps = fps.reshape(int(fps.size / num_fields), num_fields)
        fps_t = namedtuple("fps", "fps count_ranges")
        return fps_t(fps=fps, count_ranges=count_ranges)

    def delete_fps(self, ids_list):
        """Delete fps from FP db file.

        Args:
            ids_list: ids to delete list.
        Returns:
            None.
        """
        with tb.open_file(self.fp_filename, mode="a") as fp_file:
            fps_table = fp_file.root.fps
            for fp_id in ids_list:
                to_delete = [
                    row.nrow
                    for row in fps_table.where("fp_id == {}".format(str(fp_id)))
                ]
                fps_table.remove_row(to_delete[0])

    def append_fps(self, mols_source, mol_id_prop="mol_id"):
        """Appends fps to a FP db file.

        Args:
            mols_source: .smi or .sdf filename, ResultProxy or list.
        Returns:
            None.
        """
        supplier = get_mol_suplier(mols_source)
        fp_type, fp_params, rdkit_ver = self.read_parameters()
        with tb.open_file(self.fp_filename, mode="a") as fp_file:
            fps_table = fp_file.root.fps
            new_mols = []
            for mol_id, rdmol in supplier(mols_source, False, mol_id_prop=mol_id_prop):
                if not rdmol:
                    continue
                efp = rdmol_to_efp(rdmol, fp_type, fp_params)
                popcnt = py_popcount(np.array(efp, dtype=np.uint64))
                efp.insert(0, mol_id)
                efp.append(popcnt)
                new_mols.append(tuple(efp))
                if len(new_mols) == BATCH_WRITE_SIZE:
                    # append last batch < 10k
                    fps_table.append(new_mols)
                    new_mols = []
            fps_table.append(new_mols)

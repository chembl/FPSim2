from typing import Iterable as IterableType, Dict, List, Tuple, Union
from .base import BaseStorageBackend
from ..chem import (
    build_fp,
    get_mol_supplier,
    get_fp_length,
    FP_FUNC_DEFAULTS,
)
from sqlalchemy import Column, select, insert, delete, create_engine, func, MetaData
from sqlalchemy.orm import declarative_base, DeclarativeMeta
from sqlalchemy.dialects.mysql import BIGINT
from itertools import chain
import numpy as np
import rdkit
import math
import ast

BATCH_WRITE_SIZE = 32000


def build_fp_record(rdmol, fp_type, fp_params, mol_id) -> Dict:
    fp = build_fp(rdmol, fp_type, fp_params, mol_id)
    record = {f"fp_{idx}": fp for idx, fp in enumerate(fp[1:-1])}
    record.update({"mol_id": fp[0], "popcnt": fp[-1]})
    return record


def create_mapping(table_name: str, fp_length: int, base) -> DeclarativeMeta:
    clsdict = {"__tablename__": table_name}
    clsdict.update({"mol_id": Column(BIGINT(unsigned=False), primary_key=True)})
    clsdict.update(
        {
            f"fp_{idx}": Column(BIGINT(unsigned=True), primary_key=False)
            for idx in range(math.ceil(fp_length / 64))
        }
    )
    clsdict.update(
        {"popcnt": Column(BIGINT(unsigned=False), primary_key=False, index=True)}
    )
    return type(table_name, (base,), clsdict)


def create_db_table(
    mols_source: Union[str, IterableType],
    conn_url: str,
    table_name: str,
    fp_type: str,
    fp_params: dict = {},
    mol_id_prop: str = "mol_id",
    gen_ids: bool = False,
) -> None:
    # if params dict is empty use defaults
    if not fp_params:
        fp_params = FP_FUNC_DEFAULTS[fp_type]
    supplier = get_mol_supplier(mols_source)
    fp_length = get_fp_length(fp_type, fp_params)

    # define the table
    Base = declarative_base()
    comment = f"{fp_type}||{repr(fp_params)}||{rdkit.__version__}"
    FingerprintsTable = create_mapping(table_name, fp_length=fp_length, base=Base)
    FingerprintsTable.__table__.comment = comment

    # create the table
    engine = create_engine(conn_url)
    Base.metadata.create_all(engine)

    # fill the table
    with engine.begin() as conn:
        fps = []
        for mol_id, rdmol in supplier(mols_source, gen_ids, mol_id_prop=mol_id_prop):
            fp = build_fp_record(rdmol, fp_type, fp_params, mol_id)
            fps.append(fp)
            if len(fps) == BATCH_WRITE_SIZE:
                conn.execute(
                    insert(FingerprintsTable),
                    fps,
                )
                fps = []
        # append last batch < 32k
        if fps:
            conn.execute(
                insert(FingerprintsTable),
                fps,
            )


class MySQLStorageBackend(BaseStorageBackend):
    def __init__(self, conn_url: str, table_name: str = "fpsim2_fingerprints") -> None:
        super(MySQLStorageBackend, self).__init__()
        self.conn_url = conn_url

        engine = create_engine(conn_url)
        metadata = MetaData()
        metadata.reflect(engine)
        self.sqla_table = metadata.tables[table_name]

        self.in_memory_fps = True
        self.name = "mysql"
        self.fp_type, self.fp_params, self.rdkit_ver = self.read_parameters()
        self.load_fps()
        self.load_popcnt_bins()

    def read_parameters(self) -> Tuple[str, Dict[str, Dict[str, dict]], str]:
        """Reads fingerprint parameters"""
        fp_type, fp_params, rdkit_ver = self.sqla_table.comment.split("||")
        fp_params = ast.literal_eval(fp_params)
        return fp_type, fp_params, rdkit_ver

    def load_popcnt_bins(self) -> None:
        popcnt_bins = self.calc_popcnt_bins(self.fps)
        self.popcnt_bins = popcnt_bins

    def load_fps(self) -> None:
        """Loads FP db table into memory"""
        engine = create_engine(self.conn_url)
        with engine.begin() as conn:
            count = conn.scalar(select(func.count()).select_from(self.sqla_table))
            res = conn.execute(select(self.sqla_table))
            fps = np.fromiter(chain.from_iterable(res.fetchall()), dtype="<u8")
            fps = fps.reshape(count, len(self.sqla_table.columns))
        self.fps = fps[fps[:, -1].argsort()]

    def delete_fps(self, ids_list: List[int]) -> None:
        """Delete FPs from MySQL table given a list of mol ids

        Parameters
        ----------
        ids_list : list
            ids to delete.

        Returns
        -------
        None
        """
        engine = create_engine(self.conn_url)
        with engine.begin() as conn:
            conn.execute(
                delete(self.sqla_table).where(self.sqla_table.mol_id.in_(ids_list))
            )

    def append_fps(
        self, mols_source: Union[str, IterableType], mol_id_prop: str = "mol_id"
    ) -> None:
        """Appends FPs to MySQL table.

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
        engine = create_engine(self.conn_url)
        with engine.begin() as conn:
            fps = []
            for mol_id, rdmol in supplier(mols_source, gen_ids, mol_id_prop=mol_id_prop):
                fp = build_fp_record(rdmol, fp_type, fp_params, mol_id)
                fps.append(fp)
                if len(fps) == BATCH_WRITE_SIZE:
                    conn.execute(
                        insert(self.sqla_table),
                        fps,
                    )
                    fps = []
            # append last batch < 32k
            if fps:
                conn.execute(
                    insert(self.sqla_table),
                    fps,
                )

    def get_fps_chunk(self, chunk_range):
        """Loads a contiguous chunk of data from the storage into memory"""
        pass

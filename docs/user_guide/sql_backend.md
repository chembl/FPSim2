# SQLAlchemy backend

PyTables h5 file is FPSim2's primary backend for storing fingerprints, but an SQL backend is available for MySQL, PostgreSQL, or Oracle databases.

This alternative backend is particularly useful when you need to:

- Build your database incrementally
- Integrate with existing database infrastructure
- Prefer database management over file-based storage in your deployments

!!! warning
    The SQLAlchemy backend's fingerprint loading time is noticeably slower than PyTables h5 files. Once loaded, search performance is identical. For optimal loading performance with large-scale datasets, we recommend using the PyTables default backend.

## Create database table

Using a PostgreSQL for the example

```python
from FPSim2.io import create_db_table

fp_type = "Morgan"
fp_params = {"radius": 2, "fpSize": 256}
db_url = "postgresql://user:password@hostname:5432/fpsim2"
table_name = "fpsim2_fp_table"
mol_format = "smiles"

smiles_list = [
    ["Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccccc1Cl", 1],
    ["Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccc(C#N)cc1", 2],
    ["Cc1cc(-n2ncc(=O)[nH]c2=O)cc(C)c1C(O)c1ccc(Cl)cc1", 3],
    ["Cc1ccc(C(=O)c2ccc(-n3ncc(=O)[nH]c3=O)cc2)cc1", 4],
    ["Cc1cc(-n2ncc(=O)[nH]c2=O)ccc1C(=O)c1ccc(Cl)cc1", 5],
]

create_db_table(smiles_list, db_url, table_name, mol_format, fp_type, fp_params)
```


## Incremental load

You can append molecules to an existing table using the `create_db_table` function. For databases where structures are stored in the same SQL instance, you can populate the table using a SQLAlchemy CursorResult:

```python
from sqlalchemy import create_engine, text
from FPSim2.io import create_db_table

fp_type = "Morgan"
fp_params = {"radius": 2, "fpSize": 256}
db_url = "postgresql://user:password@hostname:5432/fpsim2"
table_name = "fpsim2_fp_table"
mol_format = "smiles"

sql_query = text(f"""
    SELECT
        smiles,
        mol_id
    FROM
        structure
    WHERE
        mol_id > COALESCE((
            SELECT
                MAX(mol_id)
            FROM
                {table_name}
        ), 0)
    ORDER BY
        mol_id
""")

engine = create_engine(db_url)
with engine.connect() as conn:
    cursor = conn.execute(sql_query)
    create_db_table(cursor, db_url, table_name, mol_format, fp_type, fp_params)
```

## Loading the fingerprints:

Once the fingerprints are stored in the database, load them into FPSim2Engine using the same database url and table name:

```python
from FPSim2 import FPSim2Engine

db_url = "postgresql://user:password@hostname:5432/fpsim2"
table_name = "fpsim2_fp_table"

fpe = FPSim2Engine(
    conn_url=db_url, table_name=table_name, storage_backend="sqla"
)
```

## Running a search

Running similarity searches with the SQLAlchemy backend is identical to using the PyTables backend:

```python
query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.similarity(query, threshold=0.7, metric='tanimoto', n_workers=1)
```

## Saving the table to a PyTables h5 file

You can export the fingerprints stored in SQL to the PyTables h5 file format, which is useful for sharing or distributing the data you have in SQL:

```python
fpe.save_h5("my_fps.h5")
```
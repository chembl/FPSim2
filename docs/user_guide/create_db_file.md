# Create a fingerprint database file

To create a fingerprint database file for running searches, use either:

- The `fpsim2-create-db` command line tool
- The `FPSim2.io.create_db_file` Python function

Both methods are described below.

!!! warning
    FPSim2 only supports integer molecule ids.

The fingerprints are calculated with [RDKit](https://www.rdkit.org/). Fingerprint types available are:

- [MACCSKeys](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint)
- [Morgan](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator)
- [TopologicalTorsion](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetTopologicalTorsionGenerator)
- [AtomPair](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetAtomPairGenerator)
- [RDKit](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetRDKitFPGenerator)
- [RDKitPattern](https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.PatternFingerprint)

## Using the command line
Run in parallel using .smi files as input. Example usage:

```bash
fpsim2-create-db smiles_file.smi fp_db.h5 --fp_type Morgan --fp_params '{"radius": 2, "fpSize": 256}' --processes 32
```
## Using Python
Note: When using the Python library, fingerprint calculation is single-threaded.

=== "From a .sdf file"

    ```python
    from FPSim2.io import create_db_file

    create_db_file(
        mols_source='sdf_file.sdf',
        filename='fp_db.h5',
        mol_format=None, # set to None
        fp_type='Morgan',
        fp_params={'radius': 2, 'fpSize': 256},
        mol_id_prop='mol_id'
    )
    ```

=== "From a .smi file"

    ```python
    from FPSim2.io import create_db_file

    create_db_file(
        mols_source='smiles_file.smi',
        filename='fp_db.h5',
        mol_format=None, # set to None
        fp_type='Morgan',
        fp_params={'radius': 2, 'fpSize': 256}
    )
    ```

=== "From a Python list"

    ```python
    from FPSim2.io import create_db_file

    mols = [['CC', 1], ['CCC', 2], ['CCCC', 3]]
    create_db_file(
        mols_source=mols,
        filename='fp_db.h5',
        mol_format='smiles', # required
        fp_type='Morgan',
        fp_params={'radius': 2, 'fpSize': 256}
    )
    ```


=== "From any other iterable"
    SQLAlchemy CursorResult as an example

    ```python
    from FPSim2.io import create_db_file
    from sqlalchemy import create_engine, text

    engine = create_engine('sqlite:///test/test.db')
    with engine.connect() as conn:
        sql_query = text("select molfile, mol_id from structure")
        cursor = conn.execute(sql_query)

        create_db_file(
            mols_source=cursor,
            filename='fp_db.h5',
            mol_format='molfile', # required
            fp_type='Morgan',
            fp_params={'radius': 2, 'fpSize': 256}
        )
    ```

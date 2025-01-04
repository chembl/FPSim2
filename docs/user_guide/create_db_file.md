# Create a fingerprint database file

Use the `FPSim2.io.create_db_file` function to create the fingerprint database file needed to run the searches.

!!! warning
    FPSim2 only supports integer molecule ids.

The fingerprints are calculated with [RDKit](https://www.rdkit.org/). Fingerprint types available are:

- [MACCSKeys](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint)
- [Morgan](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator)
- [TopologicalTorsion](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetTopologicalTorsionGenerator)
- [AtomPair](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetAtomPairGenerator)
- [RDKit](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetRDKitFPGenerator)
- [RDKitPattern](https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.PatternFingerprint)

# From the command line
Supports .smi files as input and runs in parallel.

```bash
fpsim2-create-db smiles_file.smi fp_db.h5 --fp_type Morgan --fp_params '{"radius": 2, "fpSize": 256}' --processes 32
```
# As a Python library
Does not run in parallel.


=== "From a .sdf file"

    ```python
    from FPSim2.io import create_db_file

    create_db_file(
        mols_source='sdf_file.sdf',
        filename='fp_db.h5',
        mol_format=None, # not required
        fp_type='Morgan',
        fp_params={'radius': 2, 'fpSize': 2048},
        mol_id_prop='mol_id'
    )
    ```

=== "From a .smi file"

    ```python
    from FPSim2.io import create_db_file

    create_db_file(
        mols_source='smiles_file.smi',
        filename='fp_db.h5',
        mol_format=None, # not required
        fp_type='Morgan',
        fp_params={'radius': 2, 'fpSize': 2048}
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
        fp_params={'radius': 2, 'fpSize': 2048}
    )
    ```


=== "From any other iterable"
    SQLAlchemy result proxy as an example

    ```python
    from FPSim2.io import create_db_file
    from sqlalchemy import create_engine, text

    engine = create_engine('sqlite:///test/test.db')
    with engine.connect() as conn:
        sql_query = text("select molfile, mol_id from structure")
        res_prox = conn.execute(sql_query)

        create_db_file(
            mols_source=res_prox,
            filename='fp_db.h5',
            mol_format='molfile', # required
            fp_type='Morgan',
            fp_params={'radius': 2, 'fpSize': 2048}
        )
    ```

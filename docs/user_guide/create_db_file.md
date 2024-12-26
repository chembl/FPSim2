# Create a fingerprint database file

Use the `FPSim2.io.backends.pytables.create_db_file` function to create the fingerprint database file needed to run the searches.

!!! warning
    FPSim2 only supports integer molecule ids.

The fingerprints are calculated with [RDKit](https://www.rdkit.org/). Fingerprint types available are:

- [MACCSKeys](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint)
- [Morgan](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator)
- [TopologicalTorsion](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetTopologicalTorsionGenerator)
- [AtomPair](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetAtomPairGenerator)
- [RDKit](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetRDKitFPGenerator)
- [RDKitPattern](https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.PatternFingerprint)

## From a .sdf file
```python
from FPSim2.io import create_db_file

create_db_file(
    mols_source='stuff.sdf',
    filename='stuff.h5',
    mol_format=None, # not required, .sdf will always use 'molfile'
    fp_type='Morgan',
    fp_params={'radius': 2, 'fpSize': 2048},
    mol_id_prop='mol_id'
)
```

## From a .smi file
```python
from FPSim2.io import create_db_file

create_db_file(
    mols_source='chembl.smi',
    filename='chembl.h5',
    mol_format=None, # not required, .smi will always use 'smiles'
    fp_type='Morgan',
    fp_params={'radius': 2, 'fpSize': 2048}
)
```

## From a Python list
```python
from FPSim2.io import create_db_file

mols = [['CC', 1], ['CCC', 2], ['CCCC', 3]]
create_db_file(
    mols_source=mols,
    filename='test/10mols.h5',
    mol_format='smiles', # required
    fp_type='Morgan',
    fp_params={'radius': 2, 'fpSize': 2048}
)
```

## From any other Python iterable like a SQLAlchemy result proxy
```python
from FPSim2.io import create_db_file
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

engine = create_engine('sqlite:///test/test.db')
s = Session(engine)
sql_query = "select molfile, mol_id from structure"
res_prox = s.execute(sql_query)

create_db_file(
    mols_source=res_prox,
    filename='test/10mols.h5',
    mol_format='molfile', # required
    fp_type='Morgan',
    fp_params={'radius': 2, 'fpSize': 2048}
)
```

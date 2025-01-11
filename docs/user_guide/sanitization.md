# Sanitization

By default, FPSim2 uses RDKit's sanitization, which is the recommended approach. However, you have two alternatives:

1. Use FPSim2's built-in partial sanitization
2. Skip sanitization completely by providing your own RDKit molecule objects to the `create_db_file` function

## Partial Sanitization

Build-in FPSim2's partial sanitization applies the following rules:

```python
def partial_sanitization(mol):
    # https://rdkit.blogspot.com/2016/09/avoiding-unnecessary-work-and.html
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol
```

Use the `--no_full_sanitization` parameter in the `fpsim2-create-db` command line:

```bash
fpsim2-create-db smiles_file.smi fp_db.h5 --no_full_sanitization --fp_type Morgan --fp_params '{"radius": 2, "fpSize": 256}' --processes 32
```

Use `full_sanitization=False` when creating databases using `create_db_file` function


```python
from FPSim2.io import create_db_file

mols = [['CC', 1], ['CCC', 2], ['CCCC', 3]]
create_db_file(
    mols_source=mols,
    filename='fp_db.h5',
    mol_format='smiles', # required
    fp_type='Morgan',
    fp_params={'radius': 2, 'fpSize': 256},
    full_sanitization=False
)
```

When running similarity searches, partial sanitization of the query molecules can be also applied using the `full_sanitization=False` flag:

```python
results = fpe.similarity(query, threshold=0.7, metric='tanimoto', full_sanitization=False, n_workers=1)
```

## Custom Sanitization

For complete control over molecule sanitization, you can provide pre-sanitized RDKit molecule objects directly to `create_db_file`. These molecules will be used as-is without any additional sanitization during fingerprint generation.

Example of custom sanitization:

```python
from FPSim2.io import create_db_file
from rdkit import Chem

mols = [['CC', 1], ['CCC', 2], ['CCCC', 3]]

def parse_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    # Apply custom sanitization steps
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_CLEANUP^Chem.SANITIZE_CLEANUPCHIRALITY)
    return mol

# Create list of [mol, id] pairs
mols = [[parse_molecule(smi), mol_id] for smi, mol_id in mols]

create_db_file(
    mols_source=mols,
    filename='fp_db.h5',
    mol_format='rdkit',  # Important: specify rdkit format
    fp_type='Morgan',
    fp_params={'radius': 2, 'fpSize': 256}
)
```

Run a search with the custom sanitization:

```python
def parse_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    # Apply custom sanitization steps
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_CLEANUP^Chem.SANITIZE_CLEANUPCHIRALITY)
    return mol

query = 'CC(=O)Oc1ccccc1C(=O)O'
results = fpe.similarity(parse_molecule(query), threshold=0.7, metric='tanimoto', full_sanitization=False, n_workers=1)
```

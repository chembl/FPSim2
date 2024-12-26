# Limitations

Due to it's simplicity FPSim2 can only use integer ids to store the fingerprints, however it can generate new ids for the provided molecules using gen_ids flag.

```python
create_db_file('mols.smi', 'mols.h5', mol_format=None, fp_type='Morgan', fp_params={'radius': 2, 'fpSize': 2048})
```

In case RDKit is not able to load a molecule, the id assigned to the molecule will be also skipped so the nth molecule in the input file will have id=n.

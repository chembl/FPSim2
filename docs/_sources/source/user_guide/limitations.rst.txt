.. _limitations:

Limitations
===========

Due to it's simplicity FPSim2 can only use integer ids to store the fingerprints, however it can generate new ids for the provided molecules using gen_ids flag.

    >>> create_db_file('mols.smi', 'mols.h5', 'Morgan', {'radius': 2, 'nBits': 2048}, gen_ids=True)

In case RDKit is not able to load a molecule, the id assigned to the molecule will be also skipped so the nth molecule in the input file will have id=n.
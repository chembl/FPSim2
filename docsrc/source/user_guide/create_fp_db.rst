.. _create:

Create a fingerprint database file
==================================

Use the :func:`~FPSim2.io.backends.pytables.create_db_file` function to create the fingerprint database file.

.. caution::
    FPSim2 only supports integer molecule ids.

The fingerprints are calculated with `RDKit <https://www.rdkit.org/>`_. Fingerprint types available are:

    - `MACCSKeys <http://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint/>`_
    - `Avalon <http://rdkit.org/docs/source/rdkit.Avalon.pyAvalonTools.html#rdkit.Avalon.pyAvalonTools.GetAvalonFP/>`_
    - `Morgan <http://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect/>`_
    - `TopologicalTorsion <http://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect/>`_
    - `AtomPair <http://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect/>`_
    - `RDKit <http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.RDKFingerprint/>`_
    - `RDKPatternFingerprint <http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.PatternFingerprint/>`_


From a .sdf file
----------------

    >>> from FPSim2.io import create_db_file
    >>> create_db_file('chembl.sdf', 'chembl.h5', 'Morgan', {'radius': 2, 'nBits': 2048}, mol_id_prop='mol_id')

From a .smi file
----------------

    >>> from FPSim2.io import create_db_file
    >>> create_db_file('chembl.smi', 'chembl.h5', 'Morgan', {'radius': 2, 'nBits': 2048})

From a Python list
------------------

    >>> from FPSim2.io import create_db_file
    >>> create_db_file([['CC', 1], ['CCC', 2], ['CCCC', 3]], 'test/10mols.h5', 'Morgan', {'radius': 2, 'nBits': 2048})


From any other Python iterable like a SQLAlchemy result proxy
-------------------------------------------------------------

.. code-block:: python

    from FPSim2.io import create_db_file
    from sqlalchemy.orm import Session
    from sqlalchemy import create_engine
    
    engine = create_engine('sqlite:///test/test.db')
    s = Session(engine)
    sql_query = "select mol_string, mol_id from structure"
    res_prox = s.execute(sql_query)
    create_db_file(res_prox, 'test/10mols.h5', 'Morgan', {'radius': 2, 'nBits': 2048})

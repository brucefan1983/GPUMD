.. _nep_dipole_out:
.. index::
   single: dipole_train.out (output file)
   single: dipole_test.out (output file)

``dipole_*.out``
================

The ``dipole_train.out`` and ``dipole_test.out`` files contain the predicted and target dipole values.

The rows depend on :ref:`atomic_v <kw_atomic_v>`:

* With ``atomic_v 0`` (the default), there is one row per configuration in the corresponding :ref:`train.xyz or test.xyz input file <train_test_xyz>`.
* With ``atomic_v 1``, there is one row per atom, in configuration order and then atom order within each configuration.

There are 6 columns.
The first 3 columns give the :math:`x`, :math:`y`, and :math:`z` dipole components calculated using the :term:`NEP` model.
The last 3 columns give the corresponding target values.

With ``atomic_v 0``, both predicted and target dipole values are divided by the number of atoms in the configuration.
With ``atomic_v 1``, the values are atomic quantities and are not divided by the number of atoms.

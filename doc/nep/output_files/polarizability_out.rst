.. _nep_polarizability_out:
.. index::
   single: polarizability_train.out (output file)
   single: polarizability_test.out (output file)

``polarizability_*.out``
========================

The ``polarizability_train.out`` and ``polarizability_test.out`` files contain the predicted and target polarizability values.

The rows depend on :ref:`atomic_v <kw_atomic_v>`:

* With ``atomic_v 0`` (the default), there is one row per configuration in the corresponding :ref:`train.xyz or test.xyz input file <train_test_xyz>`.
* With ``atomic_v 1``, there is one row per atom, in configuration order and then atom order within each configuration.

There are 12 columns.
The first 6 columns give the :math:`xx`, :math:`yy`, :math:`zz`, :math:`xy`, :math:`yz`, and :math:`zx` polarizability components calculated using the :term:`NEP` model.
The last 6 columns give the corresponding target values.

With ``atomic_v 0``, both predicted and target polarizability values are divided by the number of atoms in the configuration.
With ``atomic_v 1``, the values are atomic quantities and are not divided by the number of atoms.

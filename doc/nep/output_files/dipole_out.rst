.. _dipole_out:
.. index::
   single: dipole_train.out (output file)
   single: dipole_test.out (output file)

``dipole_*.out``
================

The ``dipole_train.out`` and ``dipole_test.out`` files contain the predicted and target dipole values.

There are :math:`N_\mathrm{c}` rows, where :math:`N_\mathrm{c}` is the number of configurations in the :ref:`train.xyz and test.xyz input files <train_test_xyz>`.

There are 6 columns.
The first 3 columns give the :math:`x`, :math:`y`, and :math:`z` dipole components calculated using the :term:`NEP` model.
The last 3 columns give the corresponding target values.

The dipole values are normalized by the number of atoms.

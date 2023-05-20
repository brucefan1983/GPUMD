.. _polarizability_out:
.. index::
   single: polarizability_train.out (output file)
   single: polarizability_test.out (output file)

``polarizability_*.out``
================

The ``polarizability_train.out`` and ``polarizability_test.out`` files contain the predicted and target polarizability values.

There are :math:`N_\mathrm{c}` rows, where :math:`N_\mathrm{c}` is the number of configurations in the :ref:`train.xyz and test.xyz input files <train_test_xyz>`.

There are 12 columns.
The first 6 columns give the :math:`xx`, :math:`yy`, :math:`zz`, :math:`xy`, :math:`yz`, and :math:`zx` polarizability components calculated using the :term:`NEP` model.
The last 6 columns give the corresponding target values.

The polarizability values are normalized by the number of atoms.

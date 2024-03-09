.. _stress_out:
.. index::
   single: stress_train.out (output file)
   single: stress_test.out (output file)

``stress_*.out``
================

The ``stress_train.out`` and ``stress_test.out`` files contain the predicted and target stress components.

There are :math:`N_\mathrm{c}` rows, where :math:`N_\mathrm{c}` is the number of configurations in the :ref:`train.xyz and test.xyz input files <train_test_xyz>`.

There are 12 columns.
The first 6 columns give the :math:`xx`, :math:`yy`, :math:`zz`, :math:`xy`, :math:`yz`, and :math:`zx` stress components calculated using the :term:`NEP` model.
The last 6 columns give the corresponding target stress components.

The stress values are in units of GPa.

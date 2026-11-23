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

For a structure without target virial or stress, a target value of -1e6 will be output to remind the user about this.

The stress values are in units of GPa.
Both predicted and target values use :math:`\mathbf{W}/V`, with diagonal components positive under compression, and contain no kinetic contribution.
This is opposite in sign to the :math:`-\mathbf{W}/V` convention for the ``stress`` field in :ref:`train.xyz and test.xyz <train_test_xyz>`.

NEP-Spin also writes 12 columns, derived from the symmetric part of each full virial,
:math:`(\mathbf{W}+\mathbf{W}^{\mathsf{T}})/(2V)`, converted to GPa.
These files are derived outputs; NEP-Spin does not read the ``stress`` input field.

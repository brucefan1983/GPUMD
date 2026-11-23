.. _virial_out:
.. index::
   single: virial_train.out (output file)
   single: virial_test.out (output file)

``virial_*.out``
================

The ``virial_train.out`` and ``virial_test.out`` files contain the predicted and target virials.

There are :math:`N_\mathrm{c}` rows, where :math:`N_\mathrm{c}` is the number of configurations in the :ref:`train.xyz and test.xyz input files <train_test_xyz>`.

For models other than NEP-Spin, there are 12 columns.
The first 6 columns give the :math:`xx`, :math:`yy`, :math:`zz`, :math:`xy`, :math:`yz`, and :math:`zx` virial components calculated using the :term:`NEP` model.
The last 6 columns give the corresponding target virials.

For a structure without target virial or stress, a target value of -1e6 will be output to remind the user about this.

The virial values are in units of eV/atom.

For NEP-Spin, the same files have 18 columns: nine predicted components followed by nine reference components, both in the row-major order of the input ``virial`` field
(:math:`xx,xy,xz,yx,yy,yz,zx,zy,zz`).
For a frame without a reference, all nine reference values are ``-1e6``.
See :ref:`stress_out` for the derived six-component stress output.

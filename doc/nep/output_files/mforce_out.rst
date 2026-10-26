.. _mforce_out_nep:
.. index::
   single: mforce_train.out (output file)
   single: mforce_test.out (output file)

``mforce_*.out``
=================

The ``mforce_train.out`` and ``mforce_test.out`` files contain the predicted
and target magnetic-force components of labeled configurations provided in
the :ref:`train.xyz and test.xyz input files <train_test_xyz>`. Configurations
without magnetic-force labels are omitted during training diagnostics.

With ``prediction 1``, ``mforce_train.out`` includes every atom in the original
``train.xyz`` order, including configurations without magnetic-force labels.
For those configurations the last three columns are zero placeholders, not
reference measurements; they must not be used to calculate a prediction error.

There are 6 columns. The first three columns are the :math:`x`, :math:`y`, and
:math:`z` magnetic-force components predicted by the :term:`NEP` model in eV/muB. The last three columns are the corresponding target magnetic
forces. Each row corresponds to one atom in an included configuration; inactive
spin-DOF types have zero predicted magnetic force.

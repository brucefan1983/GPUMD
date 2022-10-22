.. _force_out:
.. index::
   single: force_train.out (output file)
   single: force_test.out (output file)

``force_*.out``
===============

The ``force_train.out`` and ``force_test.out`` files contain the predicted and target force components of the configurations provideed in the :ref:`train.xyz and test.xyz input files <train_test_xyz>`.

There are 6 columns.
The first three columns are the :math:`x`, :math:`y`, and :math:`z` force components in units of eV/Å computed using the :term:`NEP` model.
The last three columns are the corresponding target forces.
Each row corresponds to one atom.

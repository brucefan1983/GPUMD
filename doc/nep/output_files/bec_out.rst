.. _bec_out:
.. index::
   single: bec_train.out (output file)
   single: bec_test.out (output file)

``bec_*.out``
=============

The ``bec_train.out`` and ``bec_test.out`` files contain the predicted and target Born effective charges (:term:`BEC`) of the configurations provided in the :ref:`train.xyz and test.xyz input files <train_test_xyz>`.

There are 18 columns.
The first 9 columns are the predicted :term:`BEC` components in units of e.
The last 9 columns are the corresponding target values.
The 9 columns are arranged in the order of 11, 12, 13, 21, 22, 23, 31, 32, 33. 
Each row corresponds to one atom.
The order of the atoms is consistent with the training/test dataset for the prediction mode or the full-batch training mode.
For a structure without target :term:`BEC`, a target value of 0 for each atom will be output.


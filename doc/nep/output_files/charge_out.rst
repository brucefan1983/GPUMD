.. _charge_out:
.. index::
   single: charge_train.out (output file)
   single: charge_test.out (output file)

``charge_*.out``
================

The ``charge_train.out`` and ``charge_test.out`` files contain the predicted partial charges of the configurations provided in the :ref:`train.xyz and test.xyz input files <train_test_xyz>`.

There is a single column.
Each row gives the predicted charge of one atom in units of the elementary charge e.
The order of the atoms is consistent with the training/test dataset for the prediction mode or the full-batch training mode.

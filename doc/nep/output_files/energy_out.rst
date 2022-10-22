.. _energy_out:
.. index::
   single: energy_train.out (output file)
   single: energy_test.out (output file)

``energy_*.out``
================

The ``energy_train.out`` and ``energy_test.out`` files contain the predicted and target energies for the training and test sets, respectively.
Each file contains 2 columns.
The first column gives the energy in units of eV/atom calculated using the :term:`NEP` model.
The second column gives the corresponding target energies.
Each row corresponds to the configuration at the same position in the :ref:`train.xyz and test.xyz files <train_test_xyz>`, respectively.

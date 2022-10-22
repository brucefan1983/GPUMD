.. index::
   single: nep output files

Output files
============

The ``nep`` executable produces several outout files.
The :ref:`loss.out file <loss_out>` is written in "append mode", while all other files are continuously overwritten.

The content of the :ref:`energy_train.out/energy_test.out <energy_out>`, :ref:`force_train.out/force_test.out <force_out>`, and :ref:`virial_train.out/virial_test.out <virial_out>` files are updated every 1000 steps, while the content of the other output files ar updated every 100 steps.

With the exception of the `nep.txt file <nep_txt>`, the output files contain only numbers (no text) in matrix form.
All the files are plain text files.

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: auto

   * - File name
     - Brief description
   * - :ref:`loss.out <loss_out>`
     - loss function, regularization terms, and :term:`RMSE` data as a function of generation
   * - :ref:`nep.txt <nep_txt>`
     - :term:`NEP` potential parameters
   * - :ref:`nep.restart <nep_restart>`
     - restart file 
   * - :ref:`energy_train.out <energy_out>`
     - target and predicted energies for training data set
   * - :ref:`energy_test.out <energy_out>`
     - target and predicted energies for test data set
   * - :ref:`force_train.out <force_out>`
     - target and predicted forces for training data set
   * - :ref:`force_test.out <force_out>`
     - target and predicted forces for test data set
   * - :ref:`virial_train.out <virial_out>`
     - target and predicted virials for training data set
   * - :ref:`virial_test.out <virial_out>`
     - target and predicted virials for test data set

.. toctree::
   :maxdepth: 0
   :caption: Contents

   loss_out
   nep_txt
   nep_restart
   energy_out
   force_out
   virial_out

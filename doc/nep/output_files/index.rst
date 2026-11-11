.. index::
   single: nep output files

Output files
============

The ``nep`` executable produces several output files.
The :ref:`loss.out file <loss_out>` is written in "append mode". Prediction files, ``nep.txt``, and ``nep.restart`` are overwritten when updated.

During training, the ``*_train.out`` prediction files are updated every 1000 generations.
The :ref:`loss.out <loss_out>` file and corresponding console output, :ref:`nep.txt <nep_txt>`, :ref:`nep.restart <nep_restart>`, and ``*_test.out`` prediction files are updated at the interval set by :ref:`output_interval <kw_output_interval>` (100 generations by default).
Test-set prediction files are written only when ``test.xyz`` is provided, and the available prediction files depend on the selected model.

The :ref:`nep.txt <nep_txt>` and :ref:`nep.restart <nep_restart>` files contain named records.
The :ref:`loss.out file <loss_out>` contains comment headers followed by numerical rows.
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
   * - :ref:`force_train.out <force_out_nep>`
     - target and predicted forces for training data set
   * - :ref:`force_test.out <force_out_nep>`
     - target and predicted forces for test data set
   * - :ref:`mforce_train.out <mforce_out_nep>`
     - target and predicted magnetic forces for training data set
   * - :ref:`mforce_test.out <mforce_out_nep>`
     - target and predicted magnetic forces for test data set
   * - :ref:`virial_train.out <virial_out>`
     - target and predicted virials for training data set
   * - :ref:`virial_test.out <virial_out>`
     - target and predicted virials for test data set
   * - :ref:`stress_train.out <stress_out>`
     - target and predicted stress values for training data set
   * - :ref:`stress_test.out <stress_out>`
     - target and predicted stress values for test data set
   * - :ref:`dipole_train.out <nep_dipole_out>`
     - target and predicted dipole values for training data set
   * - :ref:`dipole_test.out <nep_dipole_out>`
     - target and predicted dipole values for test data set
   * - :ref:`polarizability_train.out <nep_polarizability_out>`
     - target and predicted polarizability values for training data set
   * - :ref:`polarizability_test.out <nep_polarizability_out>`
     - target and predicted polarizability values for test data set
   * - :ref:`charge_train.out <charge_out>`
     - predicted charge values for training data set
   * - :ref:`charge_test.out <charge_out>`
     - predicted charge values for test data set
   * - :ref:`bec_train.out <bec_out>`
     - target and predicted :term:`BEC` values for training data set
   * - :ref:`bec_test.out <bec_out>`
     - target and predicted :term:`BEC` values for test data set
   * - :ref:`descriptor.out <descriptor_out>`
     - descriptor values for training data set in prediction mode

.. toctree::
   :maxdepth: 0
   :caption: Contents

   loss_out
   nep_txt
   nep_restart
   energy_out
   force_out
   mforce_out
   virial_out
   stress_out
   dipole_out
   polarizability_out
   charge_out
   bec_out
   descriptor_out

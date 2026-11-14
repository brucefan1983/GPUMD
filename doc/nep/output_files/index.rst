.. index::
   single: nep output files

Output files
============

Training writes a model, a restart state, loss history, and predictions for comparison with the reference data.
The :ref:`loss.out file <loss_out>` is written in "append mode". Prediction files, ``nep.txt``, and ``nep.restart`` are overwritten when updated.

During training, the ``*_train.out`` prediction files are updated every 1000 generations.
The :ref:`loss.out <loss_out>` file and corresponding console output, :ref:`nep.txt <nep_txt>`, :ref:`nep.restart <nep_restart>`, and ``*_test.out`` prediction files are updated at the interval set by :ref:`output_interval <kw_output_interval>` (100 generations by default).
Test-set prediction files are written only when ``test.xyz`` is provided, and the available prediction files depend on the selected model.

In :ref:`prediction mode <kw_prediction>`, evaluation results use the ``*_train.out`` names because the structures are read from ``train.xyz``.
All these files are plain text. Read their headers and the format descriptions below for column meanings and units.

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
   * - :ref:`energy_train.out <energy_out>`, :ref:`energy_test.out <energy_out>`
     - target and predicted energies for training/test data
   * - :ref:`force_train.out <force_out_nep>`, :ref:`force_test.out <force_out_nep>`
     - target and predicted forces for training/test data
   * - :ref:`mforce_train.out <mforce_out_nep>`, :ref:`mforce_test.out <mforce_out_nep>`
     - target and predicted magnetic forces for training/test data
   * - :ref:`virial_train.out <virial_out>`, :ref:`virial_test.out <virial_out>`
     - target and predicted virials for training/test data
   * - :ref:`stress_train.out <stress_out>`, :ref:`stress_test.out <stress_out>`
     - target and predicted stress values for training/test data
   * - :ref:`dipole_train.out <nep_dipole_out>`, :ref:`dipole_test.out <nep_dipole_out>`
     - target and predicted dipole values for training/test data
   * - :ref:`polarizability_train.out <nep_polarizability_out>`, :ref:`polarizability_test.out <nep_polarizability_out>`
     - target and predicted polarizability values for training/test data
   * - :ref:`charge_train.out <charge_out>`, :ref:`charge_test.out <charge_out>`
     - predicted charge values for training/test data
   * - :ref:`bec_train.out <bec_out>`, :ref:`bec_test.out <bec_out>`
     - target and predicted :term:`BEC` values for training/test data
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

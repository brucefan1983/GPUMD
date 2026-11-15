.. _nep_executable:
.. index::
   single: nep executable

nep executable
**************

The ``nep`` executable trains NEP models and evaluates existing models against reference data.

For training:

1. Prepare :ref:`train.xyz <train_test_xyz>` with the reference structures and target quantities.
   Provide ``test.xyz`` to monitor errors on a separate test set.
2. Write :ref:`nep.in <nep_in>` to specify the chemical species, model, and training settings.
   The :ref:`keyword reference <nep_input_parameters>` groups the settings by purpose.
3. Run ``nep`` in the directory containing the input files.
4. Inspect :ref:`loss.out <loss_out>` and the :doc:`prediction files <output_files/index>` to assess the model.
   The trained model is written to :ref:`nep.txt <nep_txt>`.

To evaluate an existing model, use :ref:`prediction <kw_prediction>`; in this mode, the structures to evaluate are read from ``train.xyz``.
See :ref:`nep.restart <nep_restart>` for continuing training and :ref:`fine_tune <kw_fine_tune>` for adapting a foundation model.
For molecular dynamics with a trained energy model, follow the :ref:`gpumd guide <gpumd_executable>` and load ``nep.txt`` with :ref:`potential <kw_potential>`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   input_files/index
   input_parameters/index
   output_files/index

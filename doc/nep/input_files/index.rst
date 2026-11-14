.. index::
   single: nep input files

Input files
===========

Run ``nep`` from a directory containing:

* :ref:`nep.in <nep_in>`: the model and training settings.
* :ref:`train.xyz <train_test_xyz>`: the reference structures and target quantities.

An optional :ref:`test.xyz <train_test_xyz>` provides a separate test set.
Without it, training proceeds without test-set evaluation or test-set prediction files.
For evaluation of an existing model, use :ref:`prediction mode <kw_prediction>`; the structures to evaluate are still read from ``train.xyz``.
The other input files below are needed only for the corresponding training tasks.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   nep_in
   train_test_xyz
   response_xyz
   ediff_in

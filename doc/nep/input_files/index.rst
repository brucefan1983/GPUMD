.. index::
   single: nep input files

Input files
===========

To train a :term:`NEP` model using the ``nep`` executable, prepare the :ref:`model parameters <nep_in>` (``nep.in``) and the :ref:`training dataset <train_test_xyz>` (``train.xyz``).
The :ref:`test dataset <train_test_xyz>` (``test.xyz``) is optional. If it is omitted, training proceeds without test-set evaluation or test-set prediction files.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   nep_in
   train_test_xyz
   response_xyz
   ediff_in

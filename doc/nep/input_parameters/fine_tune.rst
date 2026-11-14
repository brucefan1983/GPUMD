.. _kw_fine_tune:
.. index::
   single: fine_tune (keyword in nep.in)

:attr:`fine_tune`
=================

This keyword instructs :program:`nep` to fine tune a model starting from a foundation model.
The syntax is::

  fine_tune <nep_model_file> <nep_restart_file>

where :attr:`<nep_model_file>` is the potential file for the foundation model and :attr:`<nep_restart_file>` is the restart file for the foundation model.

The example below uses the foundation model in ``potentials/nep/nep89_20250409``.

The hyperparameters that define the architecture of the foundation model cannot be changed and must be repeated in :attr:`nep.in`.
In this ``nep.in`` template, replace ``type <your types>`` with the number and chemical symbols of the elements to train::

  # for fine-tuning
  fine_tune nep89_20250409.txt nep89_20250409.restart
  type <your types>

  # These cannot be changed:
  version    4
  zbl        2
  cutoff     6 5
  n_max      4 4
  basis_size 8 8
  l_max      4 1 1
  neuron     80

  # These can be changed:
  lambda_1   0
  lambda_e   1
  lambda_f   1
  lambda_v   1
  batch      5000
  population 50
  save_potential 1000 0 1
  generation 5000

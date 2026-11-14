.. _nep_in:
.. index::
   single: nep.in (input file)

``nep.in``
==========

This file selects the model and controls NEP training or prediction.
For the model equations, see :ref:`NEP formalism <nep_formalism>`.

File format
-----------

Write one keyword per line::

  keyword parameter_1 parameter_2 ...

Blank lines are ignored, and ``#`` starts a comment that extends to the end of the line.
Put the required :ref:`type <kw_type>` keyword first, followed by the settings to change from their defaults.
In particular, ``type`` must precede :ref:`cutoff <kw_cutoff>`, :ref:`type_weight <kw_type_weight>`, and :ref:`spin_cutoff <kw_spin_cutoff>`.
All other keywords have defaults for ordinary NEP training.

Use :ref:`model <kw_model>` to select another model family, such as ``model qnep2``.
The default model is ordinary ``nep``.
Only NEP4 is supported, so :ref:`version <kw_version>` can be omitted or set to ``4``.

Example
-------

The following input lists common settings for an ordinary Te-Pb NEP model explicitly.
Apart from the required species list, the values shown are the defaults::

  type        2 Te Pb
  version     4
  cutoff      8 4
  n_max       6 6
  basis_size  6 6
  l_max       4 1
  neuron      30
  lambda_e    1.0
  lambda_f    1.0
  lambda_v    0.1
  batch       1000
  population  50
  generation  100000

Choose the cutoffs and other model settings for the system and training data.
For Spin training, see the :ref:`spin_version example <spin_training_example>` and :ref:`Spin potential documentation <nep_spin>`.

Keywords
--------

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: auto

   * - Keyword
     - Brief description
   * - :ref:`version <kw_version>`
     - select the NEP version (currently only NEP4)
   * - :ref:`type <kw_type>`
     - number of atom types and list of chemical species
   * - :ref:`type_weight <kw_type_weight>`
     - force weights for different atom types
   * - :ref:`model <kw_model>`
     - select the model for training or prediction
   * - :ref:`flip_charge <kw_flip_charge>`
     - reverse charge-output weights when restarting charge-model training
   * - :ref:`prediction <kw_prediction>`
     - select between training and prediction (inference)
   * - :ref:`zbl <kw_zbl>`
     - outer cutoff for the universal :term:`ZBL` potential [Ziegler1985]_
   * - :ref:`spin_version <kw_spin_version>`
     - select the spin descriptor version for ``model nep_spin``
   * - :ref:`spin_cutoff <kw_spin_cutoff>`
     - one spin cutoff or one cutoff per element, bounded by the NEP radial cutoff
   * - :ref:`spin_basis_size <kw_spin_basis_size>`
     - magnetic radial basis index followed by a reserved second value fixed at zero
   * - :ref:`spin_l_max <kw_spin_l_max>`
     - maximum spatial tensor rank used by the spin descriptor
   * - :ref:`spin_compress <kw_spin_compress>`
     - number of shared compressed radial channels
   * - :ref:`spin_order <kw_spin_order>`
     - highest magnetic contraction level supplied to the neural network
   * - :ref:`spin_soc <kw_spin_soc>`
     - enable joint spin--space rotation-invariant magnetic channels
   * - :ref:`spin_type <kw_spin_type>`
     - atom types with independent spin degrees of freedom
   * - :ref:`use_typewise_cutoff_zbl <kw_use_typewise_cutoff_zbl>`
     - enable typewise cutoff radii for the ZBL part of the model
   * - :ref:`cutoff <kw_cutoff>`
     - radial (:math:`r_\mathrm{c}^\mathrm{R}`) and angular (:math:`r_\mathrm{c}^\mathrm{A}`) cutoffs
   * - :ref:`n_max <kw_n_max>`
     - radial-function index limits for radial and angular descriptors
   * - :ref:`basis_size <kw_basis_size>`
     - number of radial (:math:`N_\mathrm{bas}^\mathrm{R}`) and angular (:math:`N_\mathrm{bas}^\mathrm{A}`) basis functions
   * - :ref:`l_max <kw_l_max>`
     - expansion order for angular terms
   * - :ref:`neuron <kw_neuron>`
     - number of neurons in the hidden layer (:math:`N_\mathrm{neu}`)
   * - :ref:`lambda_1 <kw_lambda_1>`
     - weight of :math:`\mathcal{L}_1`-norm regularization term
   * - :ref:`lambda_2 <kw_lambda_2>`
     - weight of :math:`\mathcal{L}_2`-norm regularization term
   * - :ref:`lambda_e <kw_lambda_e>`
     - weight of energy loss term
   * - :ref:`lambda_f <kw_lambda_f>`
     - weight of force loss term
   * - :ref:`lambda_v <kw_lambda_v>`
     - weight of virial loss term
   * - :ref:`lambda_m <kw_lambda_m>`
     - weight of Cartesian magnetic-force loss term
   * - :ref:`lambda_m_parallel <kw_lambda_m_parallel>`
     - weight of parallel magnetic-force RMSE (requires ``lambda_m_perp``)
   * - :ref:`lambda_m_perp <kw_lambda_m_perp>`
     - weight of perpendicular magnetic-force RMSE (requires ``lambda_m_parallel``)
   * - :ref:`lambda_tau <kw_lambda_tau>`
     - weight of spin-torque loss term
   * - :ref:`lambda_spin_response <kw_lambda_spin_response>`
     - weight of grouped magnetic-response loss term
   * - :ref:`lambda_q <kw_lambda_q>`
     - weight of total-charge loss term for qNEP models
   * - :ref:`lambda_z <kw_lambda_z>`
     - weight of Born-effective-charge loss term for qNEP models
   * - :ref:`lambda_d <kw_lambda_d>`
     - weight of energy-difference loss term, with the combinations in ``ediff.in``
   * - :ref:`atomic_v <kw_atomic_v>`
     - fit atomic or global dipole/polarizability targets
   * - :ref:`lambda_shear <kw_lambda_shear>`
     - extra weight of shear-virial loss terms
   * - :ref:`force_delta <kw_force_delta>`
     - bias term that can be used to make smaller forces more accurate
   * - :ref:`batch <kw_batch>`
     - batch size for training
   * - :ref:`population <kw_population>`
     - population size used in the :term:`SNES` algorithm [Schaul2011]_
   * - :ref:`generation <kw_generation>`
     - number of generations used by the :term:`SNES` algorithm [Schaul2011]_
   * - :ref:`seed <kw_seed>`
     - random seed for the :term:`SNES` algorithm
   * - :ref:`save_potential <kw_save_potential>`
     - periodically save model checkpoints and, optionally, restart checkpoints
   * - :ref:`output_interval <kw_output_interval>`
     - interval between regular training-status and model/test-set outputs
   * - :ref:`output_descriptor <kw_output_descriptor>`
     - output normalized descriptors in prediction mode
   * - :ref:`fine_tune <kw_fine_tune>`
     - fine tune a model starting from a foundation model
   * - :ref:`import_q_scaler <kw_import_q_scaler>`
     - import descriptor normalization factors from an existing ``nep.txt``

Consistency with model files already present
--------------------------------------------

A training run writes both :ref:`nep.txt <nep_txt>` and :ref:`nep.restart <nep_restart>` every :ref:`output_interval <kw_output_interval>` generations.

Each mismatch is reported on its own line, naming the keyword, the value used by the current run, whether that value was given in :attr:`nep.in` or is a default, and the value found in the file, for example::

  The model in nep.in is inconsistent with nep.txt:
      basis_size_radial: nep.in gives 6 (default), nep.txt gives 8.
      basis_size_angular: nep.in gives 6 (default), nep.txt gives 8.

The same comparison is applied to the :ref:`nep.txt <nep_txt>` read by the :ref:`import_q_scaler keyword <kw_import_q_scaler>`, where a mismatch is always an error.

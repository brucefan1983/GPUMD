.. _nep_in:
.. index::
   single: nep.in (input file)

``nep.in``
==========

This file specifies hyperparameters used for training neuroevolution potential (:term:`NEP`) models, the functional form of which is outlined :ref:`here <nep_formalism>`.
The :term:`NEP` approach was proposed in [Fan2021]_ (NEP1) and later improved in [Fan2022a]_ (NEP2), [Fan2022b]_ (NEP3), and [Song2024]_ (NEP4).
Currently, only NEP4 is supported; the :ref:`version keyword <kw_version>` therefore only accepts ``4``.

File format
-----------

In this input file, blank lines and lines starting with :attr:`#` are ignored.
One can thus write comments after :attr:`#`.

All other lines need to be of the following form::
  
  keyword parameter_1 parameter_2 ...
 
Keywords can appear in any order with the exception of the :ref:`type_weight keyword <kw_type_weight>`, which cannot appear before the :ref:`type keyword <kw_type>`. 

The :ref:`type keyword <kw_type>` does not have default parameters and *must* be set.
All other keywords have default values.

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
   * - :ref:`model_type <kw_model_type>`
     - select to train potential, dipole, or polarizability
   * - :ref:`charge_mode <kw_charge_mode>`
     - select the charge mode for a potential model
   * - :ref:`prediction <kw_prediction>`
     - select between training and prediction (inference)
   * - :ref:`zbl <kw_zbl>`
     - outer cutoff for the universal :term:`ZBL` potential [Ziegler1985]_
   * - :ref:`spin_version <kw_spin_version>`
     - enable the Spin3 descriptor
   * - :ref:`spin_cutoff <kw_spin_cutoff>`
     - one spin cutoff or one cutoff per element, bounded by the NEP radial cutoff
   * - :ref:`spin_basis_size <kw_spin_basis_size>`
     - magnetic radial basis index followed by a reserved second value fixed at zero
   * - :ref:`spin_l_max <kw_spin_l_max>`
     - maximum spatial tensor rank used by the spin descriptor
   * - :ref:`spin_compress <kw_spin_compress>`
     - number of shared compressed radial channels
   * - :ref:`spin_order <kw_spin_order>`
     - highest Spin3 contraction level supplied to the neural network
   * - :ref:`spin_soc <kw_spin_soc>`
     - enable joint spin--space rotation-invariant Spin3 channels
   * - :ref:`spin_type <kw_spin_type>`
     - atom types with independent spin degrees of freedom
   * - :ref:`use_typewise_cutoff_zbl <kw_use_typewise_cutoff_zbl>`
     - enable typewise cutoff radii for the ZBL part of the model
   * - :ref:`cutoff <kw_cutoff>`
     - radial (:math:`r_\mathrm{c}^\mathrm{R}`) and angular (:math:`r_\mathrm{c}^\mathrm{A}`) cutoffs
   * - :ref:`n_max <kw_n_max>`
     - size of radial (:math:`n_\mathrm{max}^\mathrm{R}`) and angular (:math:`n_\mathrm{max}^\mathrm{A}`) basis
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
     - fit atomic or global virial
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
   * - :ref:`initial_para <kw_initial_para>`
     - set reference parameters used to compute descriptor normalization

Consistency with model files already present
--------------------------------------------

A training run writes both :ref:`nep.txt <nep_txt>` and :ref:`nep.restart <nep_restart>` every :ref:`output_interval <kw_output_interval>` generations.

Each mismatch is reported on its own line, naming the keyword, the value used by the current run, whether that value was given in :attr:`nep.in` or is a default, and the value found in the file, for example::

  The model in nep.in is inconsistent with nep.txt:
      basis_size_radial: nep.in gives 6 (default), nep.txt gives 8.
      basis_size_angular: nep.in gives 6 (default), nep.txt gives 8.

The same comparison is applied to the :ref:`nep.txt <nep_txt>` read by the :ref:`import_q_scaler keyword <kw_import_q_scaler>`, where a mismatch is always an error.

Example
-------
Here is an example :attr:`nep.in` file using all the default parameters::
  
  type       	2 Te Pb   # this is a mandatory keyword
  version       4         # the only option
  cutoff     	8 4       # please choose these reasonably
  n_max      	6 6       # default
  basis_size	6 6       # default
  l_max      	4 1       # default
  neuron     	30        # default
  lambda_e      1.0       # default
  lambda_f      1.0       # default
  lambda_v      0.1       # default
  batch         1000      # default
  population	50        # default
  generation	100000    # default

For Spin training, see the :ref:`spin_version example <spin_training_example>`
and the :ref:`Spin potential documentation <nep_spin>`.

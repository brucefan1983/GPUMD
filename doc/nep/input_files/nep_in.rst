.. _nep_in:
.. index::
   single: nep.in (input file)

``nep.in``
==========

This file specifies hyperparameters used for training neuroevolution potential (:term:`NEP`) models, the functional form of which is outline :ref:`here <nep_formalism>`.
The :term:`NEP` approach was proposed in [Fan2021]_ (NEP1) and later improved in [Fan2022a]_ (NEP2) and [Fan2022b]_ (NEP3).
Currently, we support NEP3 and NEP4 (to be published), which can be chosen by the :ref:`version keyword <kw_version>`.

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
     - select between NEP3 and NEP4
   * - :ref:`type <kw_type>`
     - number of atom types and list of chemical species
   * - :ref:`type_weight <kw_type_weight>`
     - force weights for different atom types
   * - :ref:`model_type <kw_model_type>`
     - select to train potential, dipole, or polarizability
   * - :ref:`prediction <kw_prediction>`
     - select between training and prediction (inference)
   * - :ref:`zbl <kw_zbl>`
     - outer cutoff for the universal :term:`ZBL` potential [Ziegler1985]_
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
   * - :ref:`lambda_2 <kw_lambda_1>`
     - weight of :math:`\mathcal{L}_2`-norm regularization term
   * - :ref:`lambda_e <kw_lambda_e>`
     - weight of energy loss term
   * - :ref:`lambda_f <kw_lambda_f>`
     - weight of force loss term
   * - :ref:`lambda_v <kw_lambda_v>`
     - weight of virial loss term
   * - :ref:`force_delta <kw_force_delta>`
     - bias term that can be used to make smaller forces more accurate
   * - :ref:`batch <kw_batch>`
     - batch size for training
   * - :ref:`population <kw_population>`
     - population size used in the :term:`SNES` algorithm [Schaul2011]_
   * - :ref:`generation <kw_generation>`
     - number of generations used by the :term:`SNES` algorithm [Schaul2011]_

Example
-------
Here is an example :attr:`nep.in` file using all the default parameters::
  
  type       	2 Te Pb # this is a mandatory keyword
  version       4       # default
  cutoff     	8 4     # default
  n_max      	4 4     # default
  basis_size	8 8     # default
  l_max      	4 2 0   # default
  neuron     	30      # default
  lambda_e      1.0     # default
  lambda_f      1.0     # default
  lambda_v      0.1     # default
  batch         1000    # default
  population	50      # default
  generation	100000  # default

The `NEP tutorial <https://github.com/brucefan1983/GPUMD/tree/master/examples/11_NEP_potential_PbTe/tutorial.ipynb>`_ illustrates the construction of a :term:`NEP` model.
More examples can be found in `this repository <https://gitlab.com/brucefan1983/nep-data>`_.

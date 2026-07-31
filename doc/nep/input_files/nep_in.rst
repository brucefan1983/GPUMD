.. _nep_in:
.. index::
   single: nep.in (input file)

``nep.in``
==========

This file specifies hyperparameters used for training neuroevolution potential (:term:`NEP`) models, the functional form of which is outline :ref:`here <nep_formalism>`.
The :term:`NEP` approach was proposed in [Fan2021]_ (NEP1) and later improved in [Fan2022a]_ (NEP2), [Fan2022b]_ (NEP3), and [Song2024]_ (NEP4).
Currently, we support NEP3 and NEP4, which can be chosen by the :ref:`version keyword <kw_version>`.

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
     - select the NEP version
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
   * - :ref:`atomic_v <kw_atomic_v>`
     - fit atomic or global virial
   * - :ref:`force_delta <kw_force_delta>`
     - bias term that can be used to make smaller forces more accurate
   * - :ref:`batch <kw_batch>`
     - batch size for training
   * - :ref:`population <kw_population>`
     - population size used in the :term:`SNES` algorithm [Schaul2011]_
   * - :ref:`generation <kw_generation>`
     - number of generations used by the :term:`SNES` algorithm [Schaul2011]_

Consistency with model files already present
--------------------------------------------

A training run writes :ref:`nep.txt <nep_txt>` every :ref:`output_interval <kw_output_interval>` generations, and :ref:`nep.restart <nep_restart>` every 100 generations, which is a fixed interval and not affected by :ref:`output_interval <kw_output_interval>`.
A later run started in the same directory therefore finds these files, and :program:`nep` checks that the model they describe is the one that :attr:`nep.in` asks for.
Both the values given explicitly in :attr:`nep.in` and the defaults filled in for the keywords that are omitted take part in this comparison.

The comparison covers everything recorded in the header of :ref:`nep.txt <nep_txt>`: the model type (:ref:`version <kw_version>`, :ref:`model_type <kw_model_type>` and :ref:`charge_mode <kw_charge_mode>`), the number of species and the species themselves in the order they are listed, the :ref:`zbl <kw_zbl>` setting and its cutoffs, the :ref:`cutoff <kw_cutoff>` values, :ref:`n_max <kw_n_max>`, :ref:`basis_size <kw_basis_size>`, :ref:`l_max <kw_l_max>` and :ref:`neuron <kw_neuron>`.
The number of rows in :ref:`nep.restart <nep_restart>` is compared against the number of parameters that :attr:`nep.in` implies.

How a mismatch is reported depends on whether the files are inputs to the run:

* If :ref:`nep.restart <nep_restart>` is present the run is a resume, and any mismatch is an error.
  Continuing would otherwise mean reading the restart state of a differently shaped model, which silently corrupts the training.
  Remove :ref:`nep.restart <nep_restart>` to start a new training run instead.
* The same applies when :ref:`prediction <kw_prediction>` is set, since :ref:`nep.txt <nep_txt>` is then the model to predict with.
* If only :ref:`nep.txt <nep_txt>` is present it is assumed to be a stale output that this run is about to overwrite.
  In this case, a mismatch is reported as a warning and the run proceeds.
  Editing :attr:`nep.in` and retraining in the same directory thus keeps working.

Each mismatch is reported on its own line, naming the keyword, the value used by the current run, whether that value was given in :attr:`nep.in` or is a default, and the value found in the file, for example::

  The model in nep.in is inconsistent with nep.txt:
      basis_size_radial: nep.in gives 6 (default), nep.txt gives 8.
      basis_size_angular: nep.in gives 6 (default), nep.txt gives 8.

The same comparison is applied to the foundation model named by the :ref:`fine_tune keyword <kw_fine_tune>` and to the :ref:`nep.txt <nep_txt>` read by the :ref:`import_q_scaler keyword <kw_import_q_scaler>`, where a mismatch is always an error.

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

The `NEP tutorial <https://github.com/brucefan1983/GPUMD/tree/master/examples/11_NEP_potential_PbTe/tutorial.ipynb>`_ illustrates the construction of a :term:`NEP` model.
More examples can be found in `this repository <https://gitlab.com/brucefan1983/nep-data>`_.

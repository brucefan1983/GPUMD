.. _nep_input_parameters:
.. index::
   single: nep input parameters

Input parameters
================

The keywords below are grouped by purpose.
See :ref:`nep.in <nep_in>` for the input format and an example training setup.

.. rubric:: Model and descriptors

.. toctree::
   :maxdepth: 1

   type
   model
   version
   cutoff
   n_max
   basis_size
   l_max
   neuron
   zbl
   use_typewise_cutoff_zbl

.. rubric:: Loss function and data weights

.. toctree::
   :maxdepth: 1

   lambda_e
   lambda_f
   lambda_v
   lambda_shear
   lambda_q
   lambda_z
   lambda_d
   lambda_1
   lambda_2
   type_weight
   force_delta
   atomic_v

.. rubric:: Optimization

.. toctree::
   :maxdepth: 1

   batch
   population
   generation
   seed

.. rubric:: Prediction and output

.. toctree::
   :maxdepth: 1

   prediction
   output_interval
   save_potential
   output_descriptor

.. rubric:: Continuing and adapting training

.. toctree::
   :maxdepth: 1

   fine_tune
   import_q_scaler
   flip_charge

.. rubric:: Spin models and magnetic losses

.. toctree::
   :maxdepth: 1

   spin_version
   spin_type
   spin_cutoff
   spin_l_max
   spin_compress
   spin_order
   spin_soc
   lambda_m
   lambda_tau
   lambda_spin_response

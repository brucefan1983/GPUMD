.. _kw_mc:
.. index::
   single: mc (keyword in run.in)

:attr:`mc`
==========

The :attr:`mc` keyword is used to carry out Monte Carlo (:term:`MC`) trial steps, usually in combination with a :term:`MD` simulation.
Three :term:`MC` ensembles are supported, including the canonical, the semi-grand canonical (:term:`SGC`), and the variance-constrained semi-grand canonical (:term:`VCSGC`) [Sadigh2012a]_ [Sadigh2012b]_ ensemble.

This keyword can only be used in combination with :term:`NEP` models.

Syntax
------

:attr:`canonical`
^^^^^^^^^^^^^^^^^
If the first parameter is :attr:`canonical`, the system will be sampled in the canonical :term:`MC` ensemble.
It can be used as follows::

    mc canonical <md_steps> <mc_trials> <T_i> <T_f> [group <grouping_method> <group_id>]

This means that :attr:`mc_trials` :term:`MC` trials are performed every :attr:`md_steps` :term:`MD` steps, while the instant temperature for the :term:`MC` ensemble changes linearly from :attr:`T_i` to :attr:`T_f`.

:attr:`sgc`
^^^^^^^^^^^
If the first parameter is :attr:`sgc`, the system will be sampled in the :term:`SGC` :term:`MC` ensemble.
It can be used as follows::

    mc sgc <md_steps> <mc_trials> <T_i> <T_f> <num_species> {<species_0> <mu_0> <species_1> <mu_1> ...} [group <grouping_method>  <group_id>]

This means that :attr:`mc_trials` :term:`MC` trials are performed every :attr:`md_steps` :term:`MD` steps, while the instant temperature for the :term:`MC` ensemble changes linearly from :attr:`T_i` to :attr:`T_f`.

:attr:`num_species` specifies the number of species that are to be included in the sampling.
It must be no less than 2 and no larger than 4.
After specifying the number of species, one needs to specify their chemical symbols (:attr:`species_i`) and chemical potentials (:attr:`mu_i`) in units of eV.
The species can be listed in arbitrary order.
Note that only the differences between the chemical potentials matter.

:attr:`vcsgc`
^^^^^^^^^^^^^
If the first parameter is :attr:`vcsgc`, the system will be sampled in the :term:`VCSGC` :term:`MC` ensemble.
It can be used in the following way::

    mc vcsgc <md_steps> <mc_trials> <T_i> <T_f> <num_species> {<species_0> <phi_0> <species_1> <phi_1> ...} kappa [group <grouping_method>  <group_id>]

This means that :attr:`mc_trials` :term:`MC` trials are performed every :attr:`md_steps` :term:`MD` steps, while the instant temperature for the :term:`MC` ensemble changes linearly from :attr:`T_i` to :attr:`T_f`.

:attr:`num_species` specifies the number of species that are to be included in the sampling.
It must be no less than 2 and no larger than 4.
After specifying the number of species, one needs to specify their chemical symbols (:attr:`species_i`) and chemical potentials (:attr:`phi_i` = :math:`\phi_i`).
The species can be listed in arbitrary order.
Next one needs to specify the (dimensionless) :attr:`kappa` parameter (:math:`\kappa`).

The :math:`\phi` and :math:`\kappa` parameters constrain the average and variance of the species concentrations, respectively.
One can usually achieve a sampling of the full composition range by varying :math:`\phi_i` between −1.2 and +1.2, which thus play a role that is equivalent to the :math:`\mu_i` parameters in the :term:`SGC` ensemble.
Typically a :math:`\kappa` value of 100 is suitable.
If the concentration fluctuations are too large (e.g., deep with miscibility gaps) one should increase this value.

The choice of parameters that we use here differs from the original papers [Sadigh2012a]_ [Sadigh2012b]_ in terms of normalization and follows the expressions in e.g., [Rahm2021]_.

General
^^^^^^^
* The listed species must be supported by the :term:`NEP` model.

* For all the :term:`MC` ensembles, there is an option to specify the grouping method :attr:`grouping_method` and the group ID :attr:`group_id` in the given grouping method, after the parameter :attr:`group`. 
  The functionality is illustrated in the example section below.

* There must be at least one listed species in the initial model system or specified group. For example, if you list Au and Cu for doing :term:`SGC` :term:`MC`, the system or the specified group must have some Au or Cu atoms (or both); otherwise the :term:`MC` trial cannot get started.

Example 1
---------

An example for sampling in the canonical ensemble is::
  
  mc canonical 100 200 500 100 group 1 3

This means

* Perform 200 :term:`MC` trials after every 100 :term:`MD` steps.
* Change the temperature for the :term:`MC` simulation linearly from 500 to 300 K, even though the temperature for the :term:`MD` ensemble is kept to be 300 K.
* Only the atoms in group 3 of grouping method 1 will be considered during :term:`MC` sampling. 

Example 2
---------

Here is an example for :term:`MC` sampling the :term:`SGC` ensemble::
  
  mc sgc 100 1000 300 300 2 Cu 0 Au 0.6

This means

* Perform 1000 :term:`MC` trials after every 100 :term:`MD` steps.
* The temperature for the :term:`MC` ensemble will be kept at 300 K.
* Only the Cu and Au atoms are involved in the :term:`MC` process. 
  The Au atoms have a chemical potential of 0.6 eV relative to the Cu atoms.

Example 3
---------

Here is an example for sampling in the :term:`VCSGC` ensemble::
  
  mc vcsgc 200 1000 500 500 2 Al -2 Ag 0 10000

This means

* Perform 1000 :term:`MC` trials after every 200 :term:`MD` steps.
* The temperature for the :term:`MC` ensemble will be kept at 500 K.
* Only the Al and Ag atoms are involved in the :term:`MC` process.
  The dimensionless :math:`\phi` parameters for Al and Ag are −2 and 0, respectively.
  The dimensionless :math:`\kappa` parameter is 10000.

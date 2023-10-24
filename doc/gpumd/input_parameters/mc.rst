.. _kw_mc:
.. index::
   single: mc (keyword in run.in)

:attr:`mc`
==========

The :attr:`mc` keyword is used to invoke Monte Carlo (:term:`MC`) simulation, usually in combination with :term:`MD` simulation. 
Three :term:`MC` ensembles are supported, including the canonical, the semi-grand canonical (:term:`SGC`), and the variance-constrained semi-grand canonical (:term:`VCSGC`) [Sadigh2012a]_ [Sadigh2012b]_ ones. 

This keyword can only be used when the potential is :term:`NEP`.

Syntax
------

:attr:`canonical`
^^^^^^^^^^^^^^^^^
If the first parameter is :attr:`canonical`, it means to use the canonical :term:`MC` ensemble.

It can be used in the following way::

    mc canonical <md_steps> <mc_trials> <T_i> <T_f> [group <grouping_method> <group_id>]

:attr:`sgc`
^^^^^^^^^^^
If the first parameter is :attr:`sgc`, it means to use the :term:`SGC` :term:`MC` ensemble.

It can be used in the following way::

    mc sgc <md_steps> <mc_trials> <T_i> <T_f> <num_species> {species_0 mu_0 species_1 mu_1 ...} [group <grouping_method>  <group_id>]

:attr:`vcsgc`
^^^^^^^^^^^
If the first parameter is :attr:`vcsgc`, it means to use the :term:`VCSGC` :term:`MC` ensemble.

It can be used in the following way::

    mc vcsgc <md_steps> <mc_trials> <T_i> <T_f> <num_species> {species_0 phi_0 species_1 phi_1 ...} kappa [group <grouping_method>  <group_id>]

* :attr:`mc_trials` :term:`MC` trials are performed every :attr:`md_steps` :term:`MD` steps.

* The instant temperature for the :term:`MC` ensemble will linearly change from attr:`T_i` to attr:`T_f`.

* :attr:`num_species` is the number of species to be involved in the :term:`SGC` or :term:`VCSGC` ensemble. It is required to be no less than 2 and no larger than 4.

* For the :term:`SGC` ensemble, after specifying the number of species to be involved, the chemical symbols and chemical potentials (in units of eV) for these species should be listed, in an arbitrary order.

* For the :term:`VCSGC` ensemble, after specifying the number of species to be involved, the chemical symbols and (dimensionless) :math:`\phi` parameters for these species should be listed, in an arbitrary order. One then needs to specify the (dimensionless) :math:`\kappa` parameter. The :math:`\phi` and :math:`\kappa` parameters constrain the average and variance of the species concentrations, respectively. (Do we need to cite papers for the exact definitions of these parameters?)

* The listed species must be supported by the :term:`NEP` model.

* For all the :term:`MC` ensembles, there is an option to specify the grouping method :atrr:`grouping_method` and the group ID :atrr:`group_id` in the given grouping method, after the parameter :atrr:`group`. See the examples below for concrete illustrations.

* There must be at least one listed species in the initial model system or specified group. For example, if you list Au and Cu for doing :term:`SGC` :term:`MC`, the system or the specified group must have some Au or Cu atoms (or both); otherwise the :term:`MC` trial cannot get started.

Example 1
---------

An example for using the canonical :term:`MC` ensemble is
  
  ensemble nvt_lan 300 300 100
  # other keywords for the run
  mc canonical 100 200 500 100 group 1 3
  run 1000000

This means that

* Will perform 200 :term:`MC` trials after every 100 :term:`MD` steps.
* The temperature for the :term:`MC` ensemble will be linearly changed from 500 to 300 K, even though the temperature for the :term:`MD` ensemble is kept to be 300 K.
* Only the atoms in group 3 of grouping method 1 will be involved in the :term:`MC` process. 

Example 2
---------

Here is an example for using the :term:`SGC` :term:`MC` ensemble:
  
  ensemble nvt_lan 300 300 100
  # other keywords for the run
  mc sgc 100 1000 300 300 2 Cu 0 Au 0.6
  run 1000000

This means that

* Will perform 1000 :term:`MC` trials after every 100 :term:`MD` steps.
* The temperature for the :term:`MC` ensemble will be kept at 300 K.
* Only the Cu and Au atoms are involved in the :term:`MC` process. The Au atoms have a chemical potential of 0.6 eV relative to the Cu atoms.

Example 3
---------

Here is an example for sampling in the :term:`VCSGC` ensemble:
  
  ensemble nvt_lan 300 300 100
  # other keywords for the run
  mc vcsgc 200 1000 500 500 2 Al -2 Ag 0 10000
  run 1000000

This means

* Perform 1000 :term:`MC` trials after every 200 :term:`MD` steps.
* The temperature for the :term:`MC` ensemble will be kept at 500 K.
* Only the Al and Ag atoms are involved in the :term:`MC` process.
  The dimensionless :math:`\phi` parameters for Al and Ag are -2 and 0, respectively.
  The dimensionless :math:`\kappa` parameter is 10000.

.. _kw_mc:
.. index::
   single: mc (keyword in run.in)

:attr:`mc`
==========

The :attr:`mc` keyword is used to carry out Monte Carlo (:term:`MC`) trial steps, usually in combination with a :term:`MD` simulation.
Three :term:`MC` ensembles are supported, including the canonical, the semi-grand canonical (:term:`SGC`), and the variance-constrained semi-grand canonical (:term:`VCSGC`) [Sadigh2012a]_ [Sadigh2012b]_ ensemble.

This keyword requires an ordinary energy NEP (``model nep``) in the
:ref:`nep.txt format <nep_txt>`.

Syntax
------

::

    mc canonical <md_steps> <mc_trials> <T_i> <T_f> [group <grouping_method> <group_id>]
    mc sgc <md_steps> <mc_trials> <T_i> <T_f> <num_species> {<species_0> <mu_0> <species_1> <mu_1> ...} [group <grouping_method>  <group_id>]
    mc vcsgc <md_steps> <mc_trials> <T_i> <T_f> <num_species> {<species_0> <phi_0> <species_1> <phi_1> ...} <kappa> [group <grouping_method>  <group_id>]

Parameters
----------

For all three forms, ``mc_trials`` trials are performed every ``md_steps`` MD steps.
The MC target temperature changes linearly from ``T_i`` to ``T_f`` during the run.

* The listed species must be supported by the :term:`NEP` model.

* For all the :term:`MC` ensembles, there is an option to specify the grouping method :attr:`grouping_method` and the group ID :attr:`group_id` in the given grouping method, after the parameter :attr:`group`.
  The functionality is illustrated in the example section below.

* There must be at least one listed species in the initial model system or specified group. For example, if you list Au and Cu for doing :term:`SGC` :term:`MC`, the system or the specified group must have some Au or Cu atoms (or both); otherwise the :term:`MC` trial cannot get started.

:attr:`canonical`
^^^^^^^^^^^^^^^^^
Canonical sampling exchanges atomic species while keeping their counts fixed.


:attr:`sgc`
^^^^^^^^^^^
SGC sampling uses species transmutations with chemical-potential differences.


:attr:`num_species` specifies the number of species that are to be included in the sampling.
It must be no less than 2 and no larger than 4.
After specifying the number of species, one needs to specify their chemical symbols (:attr:`species_i`) and chemical potentials (:attr:`mu_i`) in units of eV.
The species can be listed in arbitrary order.
Note that only the differences between the chemical potentials matter.

For a trial transmutation from species :math:`a` to species :math:`b`, GPUMD uses the acceptance probability

.. math::

   P_{a\to b}=\min\left[1,\exp\left(-\frac{\Delta U+\mu_b-\mu_a}{k_{\rm B}T}\right)\right],

where :math:`\Delta U` is the change in potential energy.
Thus, increasing the input :math:`\mu_b` makes conversion into species :math:`b` less favorable.
For a convention based on :math:`U-\sum_s\mu_s^{\rm phys}N_s`, the corresponding input values are :math:`\mu_s=-\mu_s^{\rm phys}` up to a common additive constant.

:attr:`vcsgc`
^^^^^^^^^^^^^
VCSGC sampling adds a concentration bias.


:attr:`num_species` specifies the number of species that are to be included in the sampling.
It must be no less than 2 and no larger than 4.
After specifying the number of species, one needs to specify their chemical symbols (:attr:`species_i`) and dimensionless concentration-bias parameters (:attr:`phi_i` = :math:`\phi_i`).
The species can be listed in arbitrary order.
Only differences between the :math:`\phi_i` parameters enter the acceptance criterion.
Next one needs to specify the (dimensionless) :attr:`kappa` parameter (:math:`\kappa`).

The :math:`\phi` and :math:`\kappa` parameters constrain the average and variance of the species concentrations, respectively.
For a binary system, a symmetric scan can use :math:`\phi_A=-\phi` and :math:`\phi_B=\phi`, varying the common scan parameter :math:`\phi` approximately from -1.2 to +1.2.
This changes :math:`\phi_B-\phi_A` from -2.4 to +2.4; it is distinct from varying only one species parameter while fixing the other.
The resulting compositions depend on the interactions and temperature and should be checked from the sampled concentrations.
Typically a :math:`\kappa` value of 100 is suitable.
If the concentration fluctuations are too large (e.g., deep with miscibility gaps) one should increase this value.

The parameter normalization differs from that used in the original papers [Sadigh2012a]_ [Sadigh2012b]_; see also [Rahm2021]_.

Examples
--------

Example 1
^^^^^^^^^

An example for sampling in the canonical ensemble is::

  mc canonical 100 200 500 100 group 1 3

This means

* Perform 200 :term:`MC` trials after every 100 :term:`MD` steps.
* Change the temperature for the :term:`MC` simulation linearly from 500 to 100 K.
* Only the atoms in group 3 of grouping method 1 will be considered during :term:`MC` sampling.

Example 2
^^^^^^^^^

Here is an example for :term:`MC` sampling the :term:`SGC` ensemble::

  mc sgc 100 1000 300 300 2 Cu 0 Au 0.6

This means

* Perform 1000 :term:`MC` trials after every 100 :term:`MD` steps.
* The temperature for the :term:`MC` ensemble will be kept at 300 K.
* Only the Cu and Au atoms are involved in the :term:`MC` process.
  The input parameter for Au is 0.6 eV higher than that for Cu, so the parameter contribution to the acceptance exponent favors Au-to-Cu transmutations.

Example 3
^^^^^^^^^

Here is an example for sampling in the :term:`VCSGC` ensemble::

  mc vcsgc 200 1000 500 500 2 Al -2 Ag 0 10000

This means

* Perform 1000 :term:`MC` trials after every 200 :term:`MD` steps.
* The temperature for the :term:`MC` ensemble will be kept at 500 K.
* Only the Al and Ag atoms are involved in the :term:`MC` process.
  The dimensionless :math:`\phi` parameters for Al and Ag are -2 and 0, respectively.
  The dimensionless :math:`\kappa` parameter is 10000.

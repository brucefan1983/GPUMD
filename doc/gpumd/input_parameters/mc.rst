.. _kw_mc:
.. index::
   single: mc (keyword in run.in)

:attr:`mc`
==========

This keyword is used to invoke the monte carlo (MC) simulation, usually in combination with the MD simulation. 
Three MC ensembles are supported, including the canonical, the semi-grand canonical (SGC), and the variance-constrained semi-grand canonical (VCSGC) [CITE] ones. 

Syntax
------

- SGC ensemble:

  mc sgc <md_steps> <mc_trials> <T_i> <T_f> <num_species> {species_0 mu_0 species_1 mu_1 ...} [group <grouping_method>  <group_id>]
  
- VCSGC ensemble:

  mc vcsgc <md_steps> <mc_trials> <T_i> <T_f> <num_species> {species_0 phi_0 species_1 phi_1 ...} kappa [group <grouping_method>  <group_id>]
  


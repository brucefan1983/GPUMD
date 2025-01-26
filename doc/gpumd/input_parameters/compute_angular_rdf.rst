.. _kw_compute_angular_rdf:
.. index::
   single: compute_angular_rdf (keyword in run.in)

:attr:`compute_angular_rdf`
==========================

This keyword is used to compute the angular-dependent radial distribution function (:term:`ARDF`) for all atoms or pairs of species. 
It works for classical :term:`MD`.
The results will be written to the angular_rdf.out file.

Syntax
------

This keyword is used as follows::

  compute_angular_rdf <cutoff> <r_num_bins> <angular_num_bins> <interval> [atom <i1> <i2> atom <i3> <i4> ...]

This means that the ARDF calculations will be performed every :attr:`interval` steps, with :attr:`r_num_bins` * :attr:`angular_num_bins` data points 
evenly distributed from 0 to :attr:`cutoff` (in units of Ångstrom) in terms of the distance between atom pairs and the angle between the -Pi/2 and Pi/2.

Without the optional parameters, only the total :term:`ARDF` will be calculated.

To additionally calculate the partial :term:`ARDF` for a pair of species, one can specify the types of the two species after the word "atom". 
The types 0, 1, 2, ... correspond to the species in the potential file in order. 
Currently, one can specify at most 6 pairs. 

Example
-------

   compute_angular_rdf 8.0 400 100 1000 # total ARDF every 1000 MD steps with 400*100 data up to 8 Angstrom and 100 bins in angle, 400 bins in distance
   compute_angular_rdf 8.0 400 100 1000 atom 0 0 atom 1 1 atom 0 1 # additionally calculate 3 partial ARDFs
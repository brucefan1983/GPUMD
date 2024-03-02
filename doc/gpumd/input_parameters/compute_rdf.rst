.. _kw_compute_rdf:
.. index::
   single: compute_rdf (keyword in run.in)

:attr:`compute_rdf`
===================

This keyword is used to compute the radial distribution function (:term:`RDF`) for all atoms or pairs of species. 
It works for both classical :term:`MD` and :term:`PIMD`.
The results will be written to the :ref:`rdf.out <rdf_out>` file.

Syntax
------

This keyword is used as follows::

  compute_rdf <cutoff> <num_bins> <interval> [atom <i1> <i2> atom <i3> <i4> ...]

This means that the :term:`RDF` calculations will be performed every :attr:`interval` steps, with :attr:`num_bins` data points evenly distributed from 0 to :attr:`cutoff` (in units of Ångstrom) in terms of the distance between atom pairs.

Without the optional parameters, only the total :term:`RDF` will be calculated.

To additionally calculate the partial :term:`RDF` for a pair of species, one can specify the types of the two species after the word "atom". 
The types 0, 1, 2, ... correspond to the species in the potential file in order. 
Currently, one can specify at most 6 pairs. 

Example
-------

   compute_rdf 8.0 400 1000 # total RDF every 1000 MD steps with 400 data up to 8 Angstrom
   compute_rdf 8.0 400 1000 atom 0 0 atom 1 1 atom 0 1 # additionally calculate 3 partial RDFs
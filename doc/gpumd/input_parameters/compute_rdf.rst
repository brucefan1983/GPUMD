.. _kw_compute_rdf:
.. index::
   single: compute_rdf (keyword in run.in)

:attr:`compute_rdf`
===================

This keyword is used to compute the radial distribution function (:term:`RDF`) for all atoms or pairs of species. 
It works for both classical :term:`MD` and :term:`PIMD`.
The results will be written to the :ref:`rdf.out <rdf_out>` file.

Mathematical details
--------------------

The (:term:`RDF`) is proportional to the relative probability of finding atomic pairs at a distance around :math:`r` in the system, and to a large extent, reflects the local structural information of the system. It is defined as

.. math::
   
   g (r) = \frac{V}{N^2}\frac{dN(r, dr)}{4\pi r^2 dr},

where :math:`dN (r, dr)` represents the number of particle pairs with distances between :math:`r` and :math:`r + dr`, :math:`V` is the volume of the system, and :math:`N` is the total number of particles in the system. 

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
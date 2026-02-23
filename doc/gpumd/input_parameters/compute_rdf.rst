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

The :term:`RDF` is proportional to the relative probability of finding atomic pairs at a distance around :math:`r` in the system, and to a large extent, reflects the local structural information of the system. It is defined as

.. math::
   
   g (r) = \frac{V}{N^2}\frac{dN(r, dr)}{4\pi r^2 dr},

where :math:`dN (r, dr)` represents the number of particle pairs with distances between :math:`r` and :math:`r + dr`, :math:`V` is the volume of the system, and :math:`N` is the total number of particles in the system. 

It is also possible to calculate the :term:`RDF` between different element types. In a system with :math:`N_a` atoms of type :math:`a` and :math:`N_b` atoms of type :math:`b`,  :math:`g_{ab}(r)` and :math:`g_{ba}(r)` is defined as

.. math::
   
   g_{ab}(r) = g_{ba}(r) = \frac{V}{N_a N_b} \frac{dN_{ab}(r, dr)}{4\pi r^2 dr},

where :math:`dN_{ab}(r, dr) = dN_{ba}(r, dr)` is the number of :math:`(a, b)` particle pairs with distances between :math:`r` and :math:`r + dr`. 

Syntax
------

This keyword is used as follows::

  compute_rdf <cutoff> <num_bins> <interval> 

This means that the :term:`RDF` calculations will be performed every :attr:`interval` steps, with :attr:`num_bins` data points evenly distributed from 0 to :attr:`cutoff` (in units of Ångstrom) in terms of the distance between atom pairs.

Starting from GPUMD-v4.9, there is no need to specify the atom pairs and the code will calculate the partial :term:`RDF`\s for all atom pairs.

Example
-------

   compute_rdf 8.0 400 1000 # Calculate all RDFs every 1000 MD steps with 400 data up to 8 Angstrom
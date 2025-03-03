.. _kw_compute_angular_rdf:
.. index::
   single: compute_angular_rdf (keyword in run.in)

:attr:`compute_angular_rdf`
===========================

This keyword is used to compute the angular-dependent radial distribution function (:term:`ARDF`) for all atoms or pairs of species. 
It works for classical :term:`MD`.
The results will be written to the :ref:`angular_rdf.out <angular_rdf_out>` file.

Mathematical details
--------------------

The ARDF is defined as

.. math::
   
   g (r, \theta) = \frac{n (r, \theta)}{\rho 2 r^2 \Delta r \Delta \theta},

where :math:`n (r, \theta)` is the number of pairs of atoms at a distance :math:`(r-\Delta r/2, r+\Delta r/2]` and an angle :math:`(\theta-\Delta \theta/2, \theta+\Delta \theta/2]` from each other, :math:`\rho` is the number density, :math:`r` is the distance between the atoms, :math:`\theta` is the angle between the atoms, :math:`\Delta r` is the bin width in distance, and :math:`\Delta \theta` is the bin width in angle.

Syntax
------

This keyword is used as follows::

  compute_angular_rdf <cutoff> <r_num_bins> <angular_num_bins> <interval> [atom <i1> <i2> atom <i3> <i4> ...]

This means that the ARDF calculations will be performed every :attr:`interval` steps, with :attr:`r_num_bins` data points evenly distributed from 0 to :attr:`cutoff` (in units of Ångstrom) in terms of the distance between atom pairs, and :attr:`angular_num_bins` data points evenly distributed from -Pi/2 to Pi/2 in terms of the angle.

Without the optional parameters, only the total :term:`ARDF` will be calculated.

To additionally calculate the partial :term:`ARDF` for a pair of species, one can specify the types of the two species after the word "atom". 
The types 0, 1, 2, ... correspond to the species in the potential file in order. 
Currently, one can specify at most 6 pairs. 

Example
-------

   compute_angular_rdf 8.0 400 100 1000 # total ARDF every 1000 MD steps with 400 bins in distance and 100 bins in angle up to 8 Angstrom
   compute_angular_rdf 8.0 400 100 1000 atom 0 0 atom 1 1 atom 0 1 # additionally calculate 3 partial ARDFs
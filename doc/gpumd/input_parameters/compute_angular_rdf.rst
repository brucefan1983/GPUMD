.. _kw_compute_angular_rdf:
.. index::
   single: compute_angular_rdf (keyword in run.in)

:attr:`compute_angular_rdf`
===========================

This keyword is used to compute the angular-dependent radial distribution function (:term:`ARDF`) for all atoms or pairs of species.
It works for classical :term:`MD`.
The results will be written to the :ref:`angular_rdf.out <angular_rdf_out>` file.

Syntax
------

This keyword is used as follows::

  compute_angular_rdf <cutoff> <r_num_bins> <angular_num_bins> <interval> [atom <i1> <i2> atom <i3> <i4> ...]

This means that the ARDF calculations will be performed every :attr:`interval` steps, with :attr:`r_num_bins` radial bins spanning 0 to :attr:`cutoff` (in units of Ångstrom), and :attr:`angular_num_bins` angular bins spanning :math:`-\pi` to :math:`\pi` (in radians). The output coordinates are the bin centers.

Without the optional parameters, only the total :term:`ARDF` will be calculated.

To additionally calculate the partial :term:`ARDF` for a pair of species, one can specify the types of the two species after the word "atom".
For a compacted multi-element NEP potential, the types 0, 1, 2, ... refer to the active atom types printed at initialization, which preserve the species order in the original potential file. For other potential models, they correspond to the species in the potential file in order.
Currently, one can specify at most 6 pairs.

Method
------

Let :math:`H(r,\theta)` count ordered center-neighbor pairs of distinct atoms in a radial-angular bin.
The total ARDF is

.. math::

   g(r,\theta)=\frac{H(r,\theta)}{N\rho\,\Delta V(r,\theta)},
   \qquad \rho=\frac{N}{V},

where :math:`N` is the number of atoms and :math:`V` is the simulation-cell volume.
Equivalently, the numerator can be written as the mean number of neighbors per center, :math:`n(r,\theta)=H(r,\theta)/N`, with denominator :math:`\rho\Delta V`.

The radial coordinate :math:`r` is the full three-dimensional interatomic distance.
The angular coordinate is the azimuth :math:`\theta=\operatorname{atan2}(y_j-y_i,x_j-x_i)` of the minimum-image displacement, measured from the positive :math:`x` axis in the :math:`xy` plane.
For radial bin edges :math:`r_-,r_+` and angular bin width :math:`\Delta\theta`, the exact bin volume used by the code is

.. math::

   \Delta V=\frac{4\pi}{3}(r_+^3-r_-^3)\frac{\Delta\theta}{2\pi}.

For narrow radial bins this approaches :math:`2r^2\Delta r\Delta\theta`.
Both radial and angular bins exclude their lower edge and include their upper edge.
The output coordinates are the bin centers.

For a requested pair of types :math:`a,b`, the partial ARDF is

.. math::

   g_{ab}(r,\theta)=\frac{H_{ab}(r,\theta)}{N_a\rho_b\,\Delta V},
   \qquad \rho_b=\frac{N_b}{V}.

Here, :math:`H_{ab}` counts displacements directed from type :math:`a` to type :math:`b`.
For different types, reversing the requested type order reverses the displacement direction and shifts its azimuth by :math:`\pi` modulo :math:`2\pi`.
For the same type, both directions of each pair are included; self pairs are excluded.
The reported distributions average the individually normalized sampled configurations.

Examples
--------

   compute_angular_rdf 8.0 400 100 1000 # total ARDF every 1000 MD steps with 400 bins in distance and 100 bins in angle up to 8 Angstrom
   compute_angular_rdf 8.0 400 100 1000 atom 0 0 atom 1 1 atom 0 1 # additionally calculate 3 partial ARDFs

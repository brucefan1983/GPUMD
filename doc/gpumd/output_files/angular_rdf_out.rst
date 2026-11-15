.. _angular_rdf_out:
.. index::
   single: angular_rdf.out (output file)

:file:`angular_rdf.out`
=======================

This file contains the angular-dependent radial distribution function (:term:`ARDF`) data.

File format
-----------

The file has the following columns:

1. ``radius``: The radial bin center :math:`r` (in Å)
2. ``theta``: The angular bin center :math:`\theta` (in radians)
3. ``total``: The total ARDF :math:`g(r,\theta)` for all atom pairs
4. ``type_i_j``: The partial ARDF :math:`g(r,\theta)` for atom pairs of type i and j (if specified)

For each radial bin, the angular bins are listed in increasing order.
With :math:`\Delta r=\mathtt{cutoff}/\mathtt{r\_num\_bins}` and :math:`\Delta\theta=2\pi/\mathtt{angular\_num\_bins}`, their centers are :math:`r_b=(b+1/2)\Delta r` and :math:`\theta_c=-\pi+(c+1/2)\Delta\theta`, using zero-based bin indices.
Each run appends one header and a block of ``r_num_bins * angular_num_bins`` rows, averaged over the sampled configurations.

Example
-------

For a radial bin width of 0.1 Å and 100 angular bins, the file starts as follows (illustrative zero values)::

    #radius theta total type_0_0 type_1_1 type_0_1
    0.05000 -3.11018 0.00000 0.00000 0.00000 0.00000
    0.05000 -3.04734 0.00000 0.00000 0.00000 0.00000
    ...

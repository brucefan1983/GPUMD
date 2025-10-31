.. _angular_rdf_out:
.. index::
   single: angular_rdf.out (output file)

:file:`angular_rdf.out`
=======================

This file contains the angular-dependent radial distribution function (:term:`ARDF`) data.

File Format
-----------

The file has the following columns:

1. ``radius``: The radial distance r (in Å)
2. ``theta``: The angle :math:`\theta` (in radians, from -:math:`\pi` to :math:`\pi`)
3. ``total``: The total ARDF :math:`g(r,\theta)` for all atom pairs
4. ``type_i_j``: The partial ARDF :math:`g(r,\theta)` for atom pairs of type i and j (if specified)

For each radius value, there will be multiple rows corresponding to different angle values.

Example
-------

Here is an example of the file content::

    #radius theta total type_0_0 type_1_1 type_0_1
    0.05 -3.14159 0.00000 0.00000 0.00000 0.00000
    0.05 -3.07959 0.00000 0.00000 0.00000 0.00000
    ...

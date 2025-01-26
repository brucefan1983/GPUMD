.. _rdf_out:
.. index::
   single: angular_rdf.out (output file)

``angular_rdf.out``
===================

This file contains the angular-dependent radial distribution function (:term:`ARDF`).
It is generated when invoking the :ref:`compute_angular_rdf keyword <kw_compute_angular_rdf>`.

File format
-----------
The data in this file are organized as follows:

* column 1: radius (in units of Å)
* column 2: angle (in units of radian)
* column 3: The (:term:`ARDF`) for the whole system
* column 4 and more: The (:term:`ARDF`) for specific atom pairs if specified

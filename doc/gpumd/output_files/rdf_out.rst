.. _rdf_out:
.. index::
   single: rdf.out (output file)

``rdf.out``
===========

This file contains the radial distribution function (:term:`RDF`).
It is generated when invoking the :ref:`compute_rdf keyword <kw_compute_rdf>`.

File format
-----------
The data in this file are organized as follows:

* column 1: radius (in units of Å)
* column 2: The (:term:`RDF`) for the whole system
* column 3 and more: The (:term:`RDF`) for specific atom pairs if specified

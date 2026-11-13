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

* column 1: radial bin center (in units of Å)
* column 2: The (:term:`RDF`) for the whole system
* column 3 and more: Partial RDFs for all distinct element pairs, including same-element pairs.

The header starts with ``#radius total`` and lists the element pairs in column order.
Each run appends a new header and data block containing the average over its sampled configurations.
The bin centers, pair counting, and normalization are defined under :ref:`compute_rdf <kw_compute_rdf>`.

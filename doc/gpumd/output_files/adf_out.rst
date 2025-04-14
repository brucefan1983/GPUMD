.. _adf_out:
.. index::
   single: adf.out (output file)

``adf.out``
===========

This file contains the Angular Distribution Function (:term:`ADF`). 
It is generated when the :ref:`compute_adf keyword <kw_compute_adf>` is used.

File Format
-------------
For global ADF, the data in this file are organized as follows:

- Column 1: Angles (in degrees)
- Column 2: The (:term:`ADF`) for the entire system

For local ADF, the data are organized as follows:

- Column 1: Angles (in degrees)
- The next Ntriples columns: The (:term:`ADF`) for each specific atom triple

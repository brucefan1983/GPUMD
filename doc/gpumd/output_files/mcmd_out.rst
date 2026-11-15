.. _mcmd_out:
.. index::
   single: mcmd.out (output file)

``mcmd.out``
============

This file records MC acceptance ratios and species concentrations.
It is generated when invoking the :ref:`mc keyword <kw_mc>`.

Results are appended to the file.

File format
-----------
The data in this file are organized as follows:

* column 1: number of :term:`MD` steps
* column 2: acceptance ratio of the :term:`MC` trials
* column 3 and more: species concentrations in the specified order

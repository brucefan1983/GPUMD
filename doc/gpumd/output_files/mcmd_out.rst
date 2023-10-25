.. _mcmd_out:
.. index::
   single: mcmd.out (output file)

``mcmd.out``
============

This file contains some data related to (:term:`MC`) simulation.
It is generated when invoking the :ref:`mc keyword <kw_mc>`.

File format
-----------
The data in this file are organized as follows:

* column 1: number of :term:`MD` steps
* column 2: acceptance ratio of the :term:`MC` trials
* column 3 and more: species concentrations in the specified order

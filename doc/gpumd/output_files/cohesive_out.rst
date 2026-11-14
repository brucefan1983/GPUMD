.. _cohesive_out:
.. index::
   single: cohesive.out (output file)

``cohesive.out``
================

This file contains the cohesive energy data produced when invoking the :ref:`compute_cohesive keyword <kw_compute_cohesive>`.
The file is overwritten each time the keyword is invoked.

File format
-----------

There are two columns:

* Column 1: the dimensionless scaling factor applied to the cell vectors selected by ``direction`` in :ref:`compute_cohesive <kw_compute_cohesive>`.
* Column 2: the total potential energy of the whole system in eV after relaxing the atomic positions at the fixed scaled cell.

Divide the second column by the number of atoms to obtain the potential energy per atom.

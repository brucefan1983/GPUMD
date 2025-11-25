.. _dpdt_out:
.. index::
   single: dpdt.out (output file)

``dpdt.out``
============

This file contains the time derivative of the polarizability and its time integration.
It is produced when invoking :ref:`compute_dpdt keyword <kw_compute_dpdt>` in the :ref:`run.in input file <run_in>`.

File format
-----------
The file is organized as follows:

* column 1: :math:`dP_x/dt` (in units of e A / fs)
* column 2: :math:`dP_y/dt` (in units of e A / fs)
* column 3: :math:`dP_z/dt` (in units of e A / fs)
* column 4: :math:`P_x` (in units of e A)
* column 5: :math:`P_y` (in units of e A)
* column 6: :math:`P_z` (in units of e A)

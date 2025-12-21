.. _dpdt_out:
.. index::
   single: dpdt.out (output file)

``dpdt.out``
============

This file contains the time derivative of the polarization and its time integration.
The integration assumes a starting value of zero for the polarization.
It is produced when invoking :ref:`compute_dpdt keyword <kw_compute_dpdt>` in the :ref:`run.in input file <run_in>`.

File format
-----------
The file is organized as follows:

* column 1: time (in units of fs)
* column 2: time derivative of the polarization in the :math:`x` direction :math:`dP_x/dt` (in units of e A / fs)
* column 3: time derivative of the polarization in the :math:`y` direction :math:`dP_y/dt` (in units of e A / fs)
* column 4: time derivative of the polarization in the :math:`z` direction :math:`dP_z/dt` (in units of e A / fs)
* column 5: polarization in the :math:`x` direction :math:`P_x` (in units of e A)
* column 6: polarization in the :math:`y` direction :math:`P_y` (in units of e A)
* column 7: polarization in the :math:`z` direction :math:`P_z` (in units of e A)

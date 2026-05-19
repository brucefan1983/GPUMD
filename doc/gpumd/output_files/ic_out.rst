.. _ic_out:
.. index::
   single: ic.out (output file)

``ic.out``
===========

This file contains the iron conductivity (:term:`IC`) based on mean-square displacement (:term:`MSD`).
It is generated when invoking the :ref:`compute_msd keyword <kw_compute_ic>`.

File format
-----------
The data in this file are organized as follows:

* column 1: correlation time (in units of ps)
* column 2: IC (in units of mS/cm) in the :math:`x` direction
* column 3: IC (in units of mS/cm) in the :math:`y` direction
* column 4: IC (in units of mS/cm) in the :math:`z` direction

Only the element selected which in potential via the arguments of the :ref:`compute_msd_keyword <kw_compute_msd>` is included in this output.

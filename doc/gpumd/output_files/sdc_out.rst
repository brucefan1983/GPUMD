.. _sdc_out:
.. index::
   single: sdc.out (output file)

``sdc.out``
===========

This file contains the velocity autocorrelation (:term:`VAC`) and self diffusion coefficient (:term:`SDC`).
It is generated when invoking the :ref:`compute_sdc keyword <kw_compute_sdc>`.

File format
-----------
The data in this file are organized as follows:

* column 1: correlation time (in units of ps)
* column 2: VAC (in units of Å\ :sup:`2`/ps\ :sup:`2`) in the :math:`x` direction
* column 3: VAC (in units of Å\ :sup:`2`/ps\ :sup:`2`) in the :math:`y` direction
* column 4: VAC (in units of Å\ :sup:`2`/ps\ :sup:`2`) in the :math:`z` direction
* column 5: SDC (in units of Å\ :sup:`2`/ps) in the :math:`x` direction
* column 6: SDC (in units of Å\ :sup:`2`/ps) in the :math:`y` direction
* column 7: SDC (in units of Å\ :sup:`2`/ps) in the :math:`z` direction

Only the group selected via the arguments of the :ref:`compute_sdc_keyword <kw_compute_sdc>` is included in this output.

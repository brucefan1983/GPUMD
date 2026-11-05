.. _mvac_out:
.. index::
   single: mvac.out (output file)

``mvac.out``
============

This file contains the normalized mass-weighted velocity autocorrelation (:term:`VAC`).
The file is generated when invoking the :ref:`compute_dos keyword <kw_compute_dos>`.

File format
-----------
The file is organized as follows:
 
* column 1: correlation time (in units of ps)
* column 2: normalized VAC :math:`C_x(t)/C_x(0)` in the :math:`x` direction (dimensionless)
* column 3: normalized VAC :math:`C_y(t)/C_y(0)` in the :math:`y` direction (dimensionless)
* column 4: normalized VAC :math:`C_z(t)/C_z(0)` in the :math:`z` direction (dimensionless)

Only the group selected via the :ref:`compute_dos keyword <kw_compute_dos>` is included in the output.

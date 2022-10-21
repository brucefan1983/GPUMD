.. _mvac_out:
.. index::
   single: mvac.out (output file)

``mvac.out``
============

This file contains the data of mass-weighted velocity autocorrelation (:term:`VAC`).
The file is generated when invoking the :ref:`compute_dos keyword <kw_compute_dos>`.

File format
-----------
The file is organized as follows:
 
* column 1: correlation time (in units of ps)
* column 2: VAC (in units of Å\ :math:`^2`\ /s\ :math:`^2`) in the math:`x` direction
* column 3: VAC (in units of Å\ :math:`^2`\ /s\ :math:`^2`) in the math:`y` direction
* column 4: VAC (in units of Å\ :math:`^2`\ /s\ :math:`^2`) in the math:`z` direction

Only the group selected via the :ref:`compute_dos keyword <kw_compute_dos>` is included in the output.

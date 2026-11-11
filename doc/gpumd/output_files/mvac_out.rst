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

Without the ``group`` option, the output covers all atoms.
With ``group <group_method> <group>``, it covers the selected group.
With ``group <group_method> -1``, a separate block of :attr:`Nc` rows is written for each group, in ascending group-index order starting from group 0.
The blocks are consecutive, with no headers or blank separators, and each group is normalized by its own zero-time VAC.

For a constant MD time step :math:`\Delta t`, each block contains the correlation times :math:`0, s\Delta t, \ldots, (N_c-1)s\Delta t`, where :math:`s` is :attr:`sample_interval`.
The results of each run are appended to the file.

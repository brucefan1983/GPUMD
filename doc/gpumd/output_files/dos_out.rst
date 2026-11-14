.. _dos_out:
.. index::
   single: dos.out (output file)

``dos.out``
===========

This file contains the data of the phonon density of states (:term:`PDOS`).
It is produced when invoking :ref:`compute_dos keyword <kw_compute_dos>` in the :ref:`run.in input file <run_in>`.

File format
-----------
The file is organized as follows:

* column 1: angular frequency :math:`\omega` (in units of THz)
* column 2: :term:`DOS` (in units of 1/THz) in the :math:`x` direction
* column 3: :term:`DOS` (in units of 1/THz) in the :math:`y` direction
* column 4: :term:`DOS` (in units of 1/THz) in the :math:`z` direction

Without the ``group`` option, the output covers all atoms; with a selected group, it covers only that group.
With ``group <group_method> -1`` in the :ref:`compute_dos keyword <kw_compute_dos>`, data are written group by group, in ascending group-index order starting from group 0.
Each group occupies :attr:`num_dos_points` consecutive rows, with no headers or blank separators.
The results of each run are appended to the file.
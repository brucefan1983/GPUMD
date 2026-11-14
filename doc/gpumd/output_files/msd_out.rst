.. _msd_out:
.. index::
   single: msd.out (output file)

``msd.out``
===========

This file contains the mean-square displacement (:term:`MSD`) and self diffusion coefficient (:term:`SDC`).
It is generated when invoking the :ref:`compute_msd keyword <kw_compute_msd>`.

File format
-----------
Each run appends a block beginning with comment lines marked by ``#``.
These record the command parameters, format version, atom and group counts, atom counts per group, initial cell vectors, sampling time interval, and column labels.
The header is followed by :attr:`Nc` data rows.
Intermediate files named ``msd_step[step].out``, produced with ``save_every``, use the same format.

The data columns are organized as follows:

* column 1: correlation time (in units of ps)
* column 2: MSD (in units of Å\ :sup:`2`) in the :math:`x` direction
* column 3: MSD (in units of Å\ :sup:`2`) in the :math:`y` direction
* column 4: MSD (in units of Å\ :sup:`2`) in the :math:`z` direction
* column 5: SDC (in units of Å\ :sup:`2`/ps) in the :math:`x` direction
* column 6: SDC (in units of Å\ :sup:`2`/ps) in the :math:`y` direction
* column 7: SDC (in units of Å\ :sup:`2`/ps) in the :math:`z` direction

For a constant MD time step :math:`\Delta t`, the correlation times are :math:`0, s\Delta t, \ldots, (N_c-1)s\Delta t`, where :math:`s` is :attr:`sample_interval`.
Without a grouping option, the output covers all atoms; with ``group``, it covers only the selected group.

With :attr:`all_groups`, one set of six MSD and SDC columns is written for each group in ascending group-index order, starting from group 0, with a shared time column.
For example, in a system with three groups, a total of 19 columns will be written.
The first column is the time, columns 2-7 are the :term:`MSD` and :term:`SDC` for group 0, columns 8-13 for group 1, and columns 14-19 for group 2.

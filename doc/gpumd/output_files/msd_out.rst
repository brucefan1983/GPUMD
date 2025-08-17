.. _msd_out:
.. index::
   single: msd.out (output file)

``msd.out``
===========

This file contains the mean-square displacement (:term:`MSD`) and self diffusion coefficient (:term:`SDC`).
It is generated when invoking the :ref:`compute_msd keyword <kw_compute_msd>`.

File format
-----------
The data in this file are organized as follows:

* column 1: correlation time (in units of ps)
* column 2: MSD (in units of Å\ :sup:`2`) in the :math:`x` direction
* column 3: MSD (in units of Å\ :sup:`2`) in the :math:`y` direction
* column 4: MSD (in units of Å\ :sup:`2`) in the :math:`z` direction
* column 5: SDC (in units of Å\ :sup:`2`/ps) in the :math:`x` direction
* column 6: SDC (in units of Å\ :sup:`2`/ps) in the :math:`y` direction
* column 7: SDC (in units of Å\ :sup:`2`/ps) in the :math:`z` direction

Only the group selected via the arguments of the :ref:`compute_msd_keyword <kw_compute_msd>` is included in this output.

In the case of :attr:`all_groups` having been specified, a set of columns ordered like above will be written for each group in the selected grouping method.
For example, in a system with three groups, a total of 19 columns will be written.
The first column is the time, columns 2-7 are the :term:`MSD` and :term:`SDC` for the first group, columns 8-13 for the second group, and columns 14-19 for group 3.

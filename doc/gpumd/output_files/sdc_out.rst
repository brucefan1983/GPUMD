.. _sdc_out:
.. index::
   single: sdc.out (output file)

``sdc.out``
===========

This file contains the velocity autocorrelation (:term:`VAC`) and self-diffusion coefficient (:term:`SDC`) calculated by :ref:`compute_sdc <kw_compute_sdc>`.
Results are appended at the end of each run that uses this command.

File format
-----------

Each run writes comment headers beginning with ``#``, followed by ``Nc`` data rows.
Each row corresponds to one correlation time.
For a constant MD time step :math:`\Delta t` and sampling interval :math:`s`, the correlation times are :math:`0,s\Delta t,\ldots,(N_c-1)s\Delta t`.
The header records the command parameters, format version, atom count, cell, sampling time interval, and column names.

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Column
     - Quantity
     - Unit
   * - 1
     - Correlation time
     - ps
   * - 2, 3, 4
     - VAC along x, y, z, respectively
     - Å\ :sup:`2`/ps\ :sup:`2`
   * - 5, 6, 7
     - SDC along x, y, z, respectively
     - Å\ :sup:`2`/ps

Notes
-----

The results cover all atoms unless a group is selected with ``compute_sdc``.
Each run contributes a separate block of averages.

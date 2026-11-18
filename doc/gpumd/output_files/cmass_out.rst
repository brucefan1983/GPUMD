.. _cmass_out:
.. index::
   single: cmass.out (output file)

``cmass.out``
=============

Each run using :ref:`compute_cmass <kw_compute_cmass>` appends a block with ``#`` comment lines for the command and column labels, followed by one row per sample.
The first sample is taken after the specified number of MD steps in that run.

.. list-table::
   :header-rows: 1

   * - Column
     - Quantity
     - Unit
   * - 1
     - Cumulative simulation time
     - ps
   * - 2–4
     - Unwrapped center-of-mass coordinates x, y, z
     - Å

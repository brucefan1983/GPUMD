.. _lsqt_velocity_out:
.. index::
   single: lsqt_velocity.out (output file)

``lsqt_velocity.out``
=====================

This file contains the group velocity from linear-scaling quantum transport (:term:`LSQT`) calculations.
It is produced when invoking the :ref:`compute_lsqt keyword <kw_compute_lsqt>` in the :ref:`run.in input file <run_in>`.

File format
-----------

* Each row contains the group velocity values (in units of m/s) for the specified energies.

* The number of columns equals the number of energy points. 

* Different rows are from different time points in the :term:`MD` simulation (supposed to be an equilibrium one), which can be averaged.

* Data within band gaps are not reliable and should not be trusted.

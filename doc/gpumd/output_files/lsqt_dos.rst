.. _lsqt_dos_out:
.. index::
   single: lsqt_dos.out (output file)

``lsqt_dos.out``
================

This file contains the electronic density of states (:term:`DOS`) from linear-scaling quantum transport (:term:`LSQT`) calculations.
It is produced when invoking the :ref:`compute_lsqt keyword <kw_compute_lsqt>` in the :ref:`run.in input file <run_in>`.

File format
-----------

* Each row contains the :term:`DOS` values (in units of states/atom/eV) for the specified energies.

* The number of columns equals the number of energy points. 

* Different rows are from different time points in the :term:`MD` simulation (supposed to be an equilibrium one), which can be averaged.

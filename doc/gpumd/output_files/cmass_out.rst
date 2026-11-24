.. _cmass_out:
.. index::
   single: cmass.out (output file)

``cmass.out``
=============

This file contains the center of mass computed by the :ref:`compute_cmass keyword <kw_compute_cmass>`.

File format
-----------

Each run appends a block beginning with ``#`` comment lines that record the command and column labels.
The header is followed by one data row per sample.

* Column 1 gives the simulation time in ps.
* Columns 2–4 give the unwrapped center-of-mass coordinates :math:`x`, :math:`y`, and :math:`z` in Å.

The first sample is taken after ``sample_interval`` MD steps in the current run.
Simulation time continues across consecutive ``run`` commands in the same GPUMD process.

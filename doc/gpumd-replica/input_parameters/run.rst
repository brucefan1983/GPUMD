.. _replica_kw_run:
.. index::
   single: run (keyword in gpumd_replica run.in)

:attr:`!run`
============

Syntax
------

::

  run <number_of_steps>

The step count must be positive. This command must appear once, at the end of ``run.in``.

* REMD: production steps per replica, excluding initial equilibration.
* PRD: correlation and parallel MD steps, excluding dephasing, minimization, and extra replay work.
* Resume: additional steps beyond the saved state.

Example
-------

::

  run 2000000

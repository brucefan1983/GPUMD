.. _replica_kw_dump_xyz:
.. index::
   single: dump_xyz (keyword in gpumd_replica run.in)

:attr:`!dump_xyz`
=================

Syntax
------

::

  dump_xyz -1 0 <interval> <filename> [velocity]

The interval must be positive. Only grouping method ``-1`` is supported; use ``0`` for the unused group index.

* REMD: writes positions every ``interval`` production steps, with optional velocities.
* PRD: writes the initial quenched basin and each accepted transition.
  The interval is validated but ignored; use ``1``. Velocities are not supported.

Example
-------

For REMD::

  dump_xyz -1 0 5000 trajectory.xyz velocity

See :ref:`replica_trajectory` for filenames and frame contents.

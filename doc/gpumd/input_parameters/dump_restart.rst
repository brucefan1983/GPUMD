.. _kw_dump_restart:
.. index::
   single: dump_restart (keyword in run.in)

:attr:`dump_restart`
====================

Write the current configuration, including atomic velocities, to :ref:`restart.xyz <restart_xyz>` for continuing a simulation.

Syntax
------

::

  dump_restart <interval>

Parameters
----------

``interval`` is a positive integer giving the number of MD steps between updates.

Examples
--------

To update the restart file every 100000 steps for a run, one can add::

  dump_restart 100000

before the :ref:`run keyword <kw_run>`.
Each update overwrites the file.
To save the configuration at the final step, choose a run length that is a multiple of the output interval.


Notes
-----
Repeat this command before each run that needs this output.

NEP-spin models also save the per-atom Cartesian ``spin:R:3`` values.

For the additional Spin data,
see :ref:`Spin restart <spin_restart>`.

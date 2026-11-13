.. _replica_restart:
.. index::
   single: gpumd output files; replica restart files

Replica restart files
=====================

At normal completion, REMD writes ``remd_restart.meta`` and ``remd_replica_<index>_restart.xyz`` for every replica.
REMD-TP writes ``remd_tp_restart.meta`` and ``remd_tp_replica_<index>_restart.xyz``, including each replica’s current box and temperature–pressure assignment.
PRD writes ``prd_restart.meta``, ``prd_replica_<index>_restart.xyz``, and ``prd_basin_restart.xyz``.
Keep the complete set: it stores atomic states, method state, and random-number-generator states.
The replica xyz files store positions in Angstrom, masses in unified atomic mass units, and velocities (``vel``) in Angstrom/fs, with 17 significant digits.
REMD restores its temperature assignments and exchange schedule.
PRD restores its basin reference, running phase, correlation progress, and physical clock.
Random-number-generator states are also restored, but a resumed run is not guaranteed to reproduce an uninterrupted trajectory bit for bit.
Exchange outcomes and event times can differ.

Example
-------

To continue REMD, keep the original input parameters and replace the ``multi_replica`` line with::

  multi_replica remd replicas 8 exchange 1000 temp 300 450 resume remd

For PRD, add ``resume prd`` to its original ``multi_replica`` line.
``run`` specifies additional steps. A prefix can include a directory, such as ``resume ../previous/remd``.
New output uses the usual filenames in the current working directory.

Caveats
-------

* Restart files with an incompatible format are rejected.
  Older replica xyz files containing an additional ``vel_natural`` property are accepted; that property supplies the velocities in GPUMD's internal unit.
* ``run.in``, ``model.xyz``, and the potential file are still required.
  Preserve atom count, species, masses, box, potential content, replica count, time step, thermostat, and method parameters.
  For charge NEP models, also keep the ``kspace`` command: the method and an explicit PPPM spacing are stored and must match.
  A restart that stores ``pppm`` without a spacing matches only a ``kspace`` command without a spacing.
* REMD resume cannot request positive ``equilibrate``.
* Restarts are written at normal completion, not periodically.
  Appended diagnostics and trajectories should correspond to the saved checkpoint.
* PRD checks for events at the end of each block.
  Ending a run can shorten the last block and add an event check, so splitting a run can change event times even without numerical roundoff.

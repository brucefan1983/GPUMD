.. _replica_restart:
.. index::
   single: gpumd_replica output files; restart

Restart files
=============

At normal completion, REMD writes ``remd_restart.meta`` and ``remd_replica_<index>_restart.xyz`` for every replica.
PRD writes ``prd_restart.meta``, ``prd_replica_<index>_restart.xyz``, and ``prd_basin_restart.xyz``.
Keep the complete set: it stores atomic states, method state, and random-number-generator states.
REMD requires restart format version 5; PRD requires version 6, which records the single-segment dephasing step count.
Older PRD restart files are rejected; start a fresh PRD run when upgrading.

Example
-------

To continue REMD, keep the original input parameters and replace the ``multi_replica`` line with::

  multi_replica remd replicas 8 exchange 1000 temp 300 450 resume remd

For PRD, add ``resume prd`` to its original ``multi_replica`` line.
``run`` specifies additional steps. A prefix can include a directory, such as ``resume ../previous/remd``.
New output uses the usual filenames in the current working directory.

Caveats
-------

* ``run.in``, ``model.xyz``, and the potential file are still required.
  Preserve atom count, species, masses, box, potential content, replica count, time step, thermostat, and method parameters.
* REMD resume cannot request positive ``equilibrate``.
* Restarts are written at normal completion, not periodically.
  Appended diagnostics and trajectories should correspond to the saved checkpoint.

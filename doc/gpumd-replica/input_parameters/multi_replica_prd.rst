.. _replica_prd:
.. index::
   single: multi_replica prd (keyword in gpumd_replica run.in)

:attr:`!multi_replica prd`
==========================

Syntax
------

::

  multi_replica prd replicas <number> [<per_gpu>] event <interval> dephase <steps> correlate <steps> distance <threshold> [quench <force_tolerance> <max_steps>] [max_dephase_retries <number>] [resume <prefix>]

* ``replicas``: positive replica count; see :ref:`replica_kw_multi_replica` for GPU allocation.
* ``event``: positive interval in MD steps between event checks.
* ``dephase``: nonnegative number of consecutive MD steps required in the basin.
  Velocities are randomized once at the start of each attempt, not between event checks.
  Escaping replicas return to their dephasing starting configuration and retry with new velocities.
  Zero disables dephasing.
* ``correlate``: nonnegative number of consecutive steps without an escape required before dephasing and parallel dynamics.
* ``distance``: positive maximum atomic displacement threshold in Angstrom between quenched configurations, using periodic minimum images.
  Must be smaller than half the shortest periodic box thickness when periodic directions are present.
* ``quench``: positive FIRE force tolerance in eV/Angstrom and positive step limit; defaults are ``1e-4 1000``.
  Unconverged quenches stop the run.
* ``max_dephase_retries``: positive limit on dephasing trial rounds, including the first attempt; default 1000.
* ``resume``: :ref:`restart prefix <replica_restart>`.

Example
-------

::

  potential nep.txt
  time_step 1
  ensemble nvt_bdp 300 300 100
  multi_replica prd replicas 8 event 10 dephase 5000 correlate 1000 distance 0.5
  dump_xyz -1 0 1 prd_events.xyz
  run 100000

Caveats
-------

* Choose the basin threshold, dephasing, and correlation times for the system of interest.
  The displacement test does not align translations, rotations, or atom permutations.
* ``dephase`` accepts one step count. The former two-argument syntax is not supported.
  A single longer segment is not equivalent to repeated shorter segments with velocity refreshes.
* A detected escape triggers step-by-step block replay; simultaneous escapes select the lowest replica index.
  An escape followed by a return before a block-end check can be missed. Use ``event 1`` to check every step.
* ``run`` counts correlation and parallel MD steps, not accelerated physical time; see :ref:`replica_kw_run`.

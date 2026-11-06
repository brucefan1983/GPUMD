.. _replica_prd:
.. index::
   single: multi_replica prd (keyword in gpumd_replica run.in)

:attr:`!multi_replica prd`
==========================

This command selects parallel replica dynamics (PRD) [Voter1998]_.
The discrete-time clock update follows [Aristoff2014]_; see :ref:`replica_prd_events`.

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
  Convergence requires the largest absolute force component among the :math:`3N` force components to be smaller than the tolerance.
  Unconverged quenches stop the run.
* ``max_dephase_retries``: positive limit on dephasing trial rounds, including the first attempt; default 1000.
* ``resume``: :ref:`restart prefix <replica_restart>`.
Batched integration
-------------------

When a GPU holds at least two replicas and at most 20000 atoms in total (replicas times atoms per
replica), its replicas are integrated as one concatenated system, with one kernel launch per stage
and step instead of one per replica. This is faster for many small replicas; for larger systems the
replicas are integrated one by one. Results are the same either way. Batching needs an energy NEP
model whose box uses the large-box path; otherwise the replicas are integrated one by one. The
choice is printed at the start of the run (``batched integration: on`` or ``off`` with the reason),
and a line of batch statistics is printed at the end. Batching covers the MD steps of dephasing,
parallel dynamics and correlation; FIRE quenches remain per replica.

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
* Escapes are detected only at block ends, for every replica alike: an escape followed by a return within a block is not an event.
  Simultaneous escapes select the lowest replica index, and the physical clock has a resolution of ``event`` steps.
  Use ``event 1`` to check every step; a larger interval saves quenches and should be much shorter than the mean escape time.
* ``run`` counts correlation and parallel MD steps, not accelerated physical time; see :ref:`replica_kw_run`.

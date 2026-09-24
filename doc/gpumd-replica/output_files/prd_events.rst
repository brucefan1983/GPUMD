.. _replica_prd_events:
.. index::
   single: gpumd_replica output files; prd_events

``prd_events.out``
==================

One row is written per accepted transition, with columns::

  dynamics_step clock_step physical_time_fs event correlated coincident replica search_steps max_displacement_A quenched_potential_eV quenched_fmax_eV_per_A quench_steps

* ``dynamics_step``: accumulated correlation and parallel MD steps.
* ``clock_step``: accumulated physical clock; ``physical_time_fs`` is this value multiplied by the MD time step in fs.
* ``event``: event count, starting at 1.
* ``correlated``: 1 during correlation, 0 during parallel dynamics.
* ``coincident``: number of replicas escaping at the selected step.
* ``replica``: selected replica index, starting at zero.
* ``search_steps``: steps searched within the current block up to the event.
* ``max_displacement_A``: maximum displacement from the previous quenched basin, in Angstrom.
* The final three columns give the selected quenched configuration's total potential energy (eV), maximum force (eV/Angstrom), and minimization steps.

For R replicas, a parallel block of s steps adds R*s clock steps if there is no escape, or R*(s-1)+r+1 if replica r escapes at step s.
The escape increment follows the discrete-time ParRep formula in [Aristoff2014]_, with the one-based replica index K=r+1.
Correlation adds one clock step per MD step; dephasing adds none.

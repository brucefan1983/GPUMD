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
* ``search_steps``: length of the block in which the escape was detected.
* ``max_displacement_A``: maximum displacement from the previous quenched basin, in Angstrom.
* The final three columns give the selected quenched configuration's total potential energy (eV), largest absolute force component among the :math:`3N` force components (eV/Angstrom), and minimization steps.

Escapes are tested on the quenched state at the end of each block of s steps (s = ``event`` interval).
For R replicas, a parallel block adds R*s clock steps if there is no escape, or s*(r+1) if the lowest escaped replica index is r.
This is the discrete-time ParRep formula in [Aristoff2014]_ for the chain observed every s steps, with the one-based replica index K=r+1.
With ``event 1`` the clock is step-resolved.
Correlation adds one clock step per MD step; dephasing adds none.

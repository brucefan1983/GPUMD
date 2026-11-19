.. _kw_multi_replica_remd:
.. _replica_remd:
.. index::
   single: multi_replica remd (keyword in run.in)

:attr:`multi_replica` (REMD)
============================

The ``remd`` mode of :ref:`multi_replica <kw_multi_replica>` runs replica-exchange
molecular dynamics [Sugita1999]_. The ``ensemble`` command selects NVT temperature
exchange or NPT temperature-pressure exchange [Okabe2001]_.

Syntax
------

NVT temperature exchange::

  ensemble nvt_bdp <T_min> <T_max> <tau_T>
  multi_replica remd replicas <number> [<per_gpu>] exchange <interval>
                temp <T_min> <T_max> [spacing <geometric|linear>]
                [equilibrate <steps>] [resume <prefix>]

Alternatively, ``temp <filename>`` reads one temperature for each replica.

.. _replica_remd_npt:

NPT temperature-pressure exchange::

  ensemble npt_scr <T_min> <T_max> <tau_T> <P_min> <P_max> <bulk_modulus> <tau_p>
                   [couple <dirs>] [pcouple_interval <steps>]
  multi_replica remd replicas <number> [<per_gpu>] exchange <interval>
                temp <T_min> <T_max> <M_T>
                pressure <P_min> <P_max> <M_P>
                [spacing <geometric|linear>] [equilibrate <steps>] [resume <prefix>]

For NPT, ``temp <filename>`` and ``pressure <filename>`` read the corresponding
ladders. ``pressure_grid <filename>`` replaces ``pressure`` for a sheared grid.
This replica-specific ``npt_scr`` syntax describes the exchange grid endpoints;
it does not use the pressure-component syntax of ordinary MD.

Parameters
----------

* ``number``: at least two replicas; see :ref:`kw_multi_replica` for GPU allocation.
* ``exchange``: positive number of MD steps between exchange rounds.
* ``temp``: positive, finite, strictly increasing temperatures in K.
  NVT ranges contain ``number`` temperatures. NPT ranges explicitly specify ``M_T``.
  ``spacing`` is ``geometric`` (default) or ``linear`` for generated temperatures.
* ``pressure``: finite pressures in GPa, linearly spaced when given as a range.
  Negative pressures are allowed. NPT requires ``number = M_T * M_P``.
  Either count may be one, in which case its endpoints must be equal.
* Ladder files contain whitespace-separated numbers; ``#`` starts a comment.
* ``pressure_grid``: one nonempty row per temperature, with the same number
  ``M_P`` of strictly increasing pressures on every row. Rows may differ, so a
  temperature-direction exchange can change both temperature and pressure.
  The ensemble pressure endpoints must equal the minimum and maximum over the grid.
* ``tau_T``: BDP thermostat relaxation time divided by the MD time step, at least 1.
  Ensemble temperatures must match the ladder endpoints and do not define a ramp.
* ``bulk_modulus``: positive finite estimate in GPa. It controls SCR's response,
  without changing its target equilibrium distribution in the continuous-time limit.
  Finite-step Euler sampling errors can depend on this value.
* ``tau_p``: positive finite barostat relaxation time in MD steps, at least twice
  ``pcouple_interval``.
* ``couple``: periodic axes scaled together, default ``xyz``. Use ``xy`` for
  lateral pressure, ``z`` for axial pressure, or another nonrepeating combination
  of ``x``, ``y`` and ``z``. The other box lengths stay fixed.
* ``pcouple_interval``: positive number of MD steps between SCR updates, default 10.
  Larger intervals reduce host synchronization and neighbor-list rebuilding, but
  increase the Euler barostat step. Assess performance, sampling error and stability
  together with ``tau_p`` and the MD time step.
* ``equilibrate``: independent steps at each replica's assigned conditions before
  production; default 0. These steps do not count toward ``run``.
* ``resume``: :ref:`restart prefix <replica_restart>`, normally ``remd`` for both
  NVT and NPT. It cannot be combined with positive ``equilibrate``.

NVT rejects ``pressure`` and ``pressure_grid``. NPT requires one pressure source.
See :ref:`kw_multi_replica` for the common input commands and trajectory options.

Output
------

NVT keeps its existing :ref:`temperature-exchange output <replica_output_files>`.
NPT uses :ref:`extended temperature-pressure diagnostics <replica_remd_npt_output>`
because its condition table and exchange columns differ.
Both use the common :ref:`trajectory <replica_trajectory>` and
:ref:`restart <replica_restart>` mechanisms. Output columns, units and header
fields are defined on those output pages.

Examples
--------

Existing NVT inputs remain valid::

  potential nep.txt
  time_step 0.5
  ensemble nvt_bdp 300 450 100
  multi_replica remd replicas 8 exchange 1000 temp 300 450 equilibrate 200000
  dump_xyz -1 0 5000 trajectory.xyz velocity
  run 2000000

NPT exchange on a 6-temperature by 4-pressure grid::

  potential nep.txt
  time_step 0.5
  ensemble npt_scr 250 300 100 0 0.15 2.0 1000
  multi_replica remd replicas 24 exchange 200 temp 250 300 6 pressure 0 0.15 4 equilibrate 20000
  dump_xyz -1 0 2000 trajectory.xyz velocity
  run 200000

For pressure-only exchange, use a single temperature, for example
``temp 300 300 1 pressure 0 1 8`` with eight replicas and matching ensemble endpoints.

Notes
-----

* NVT supports energy NEP and qNEP models; its existing thermostat convention is
  unchanged. NPT currently supports energy NEP models, orthogonal boxes and at
  least two atoms. It initializes canonical velocities with zero total linear
  momentum and uses ``3N-3`` thermal degrees of freedom.
* NPT uses BDP [Bussi2007b]_ and SCR [Bernetti2020]_. SCR evolves ``ln(V)`` with
  no additional ``k_B T / V`` drift. Pressure is measured after BDP rescaling;
  a pending box update is applied before the next MD step. Accepted temperature
  exchanges refresh its kinetic pressure while preserving the pending update.
  This is the Euler SCR integrator, with finite-step sampling error.
* NVT exchanges alternate between even and odd neighboring temperature pairs.
  NPT labels are ``p_index * M_T + t_index``; rounds cycle through even temperature,
  even pressure, odd temperature and odd pressure pairs. Dimensions of size one
  are omitted; a dimension of size two always uses its sole pair.
* Exchanges swap conditions. Positions and boxes stay with their replica;
  temperature changes rescale velocities by ``sqrt(T_new / T_old)``. The NPT
  log acceptance ratio is

  .. math::

     (\beta_a-\beta_b)(E_i-E_j)
       +(\beta_a P_a-\beta_b P_b)(V_i-V_j).

  Here ``E`` is potential energy and pressures are converted to energy per volume.
  Collect equilibrium statistics by condition label, rather than replica index.
* NPT boxes evolve independently, so :ref:`batched integration <replica_batched_integration>`
  is disabled. Pressure spacing should be chosen to obtain sufficient exchange acceptance.
* A length change exceeding 5% in one SCR update stops the run with replica,
  step, volume, pressure and scale diagnostics. Restart files are written only at
  normal completion. Basic continuation does not promise bitwise-identical
  trajectories or exchange outcomes after splitting a run.

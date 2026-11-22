.. _replica_remd_npt_output:

NPT REMD output files
=====================

:ref:`multi_replica remd <replica_remd_npt>` with ``ensemble npt_scr`` uses extended
diagnostic files to keep the existing NVT column layouts unchanged. Steps, labels,
replica indices, grid indices and counts are dimensionless integers; indices start
at zero. Condition labels are ``p_index * M_T + t_index``.
Collect equilibrium statistics by condition label, because replicas change conditions.

``remd_npt_conditions.out``
--------------------------

Written once and overwritten on resume. Full column order::

  label temperature_index pressure_index temperature pressure

Temperature is in K; pressure is in GPa. This table maps every condition label
to its grid position and target conditions.

``remd_npt_exchange.out``
------------------------

One row per attempted pair, describing states before swapping::

  step direction replica_i replica_j label_a label_b T_a T_b P_a P_b E_i E_j V_i V_j log_acceptance_ratio acceptance_probability accepted

``direction`` is ``T`` or ``P``. Temperatures are in K, pressures in GPa, total
potential energies in eV and volumes in Angstrom cubed. The log ratio and
probability are dimensionless; ``accepted`` is 0 or 1.

``remd_npt_status.out``
----------------------

Written initially and at exchange, trajectory and final boundaries, after any
exchange. Column order::

  step condition_label_of_replica_0 condition_label_of_replica_1 ...

``remd_npt_thermo.out``
----------------------

Column order is one production step followed by three columns per replica::

  step condition_label_0 potential_energy_0 volume_0 condition_label_1 potential_energy_1 volume_1 ...

Potential energies are totals in eV; volumes are in Angstrom cubed. Values describe
the current configuration after any exchange, before a pending SCR update.
Rows are written initially and at exchange, trajectory-output, progress-report
and final advancement boundaries. Sampling intervals can be nonuniform. Use the
step column when selecting or time-weighting samples.

``remd_npt_pairs.out``
---------------------

One row per neighboring condition pair; overwritten at normal completion. Columns::

  direction label_a label_b T_a T_b P_a P_b attempts accepted acceptance_ratio average_probability

Temperatures are in K and pressures in GPa. ``acceptance_ratio`` is accepted
attempts divided by attempts; ``average_probability`` is the mean Metropolis
probability over attempts. Both are dimensionless and written as zero when the
pair has not been attempted. They appear in this order, matching the data writer.

Exchange, status and thermo files append on resume. Trajectories and restart use
the common :ref:`replica trajectory <replica_trajectory>` and
:ref:`replica restart <replica_restart>` files.
